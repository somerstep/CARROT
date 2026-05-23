import numpy as np
import argparse
import os
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import Dataset
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback, get_scheduler, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss
from constants import *

# Examples:
#   python train_and_infer.py --dataset routerbench
#   python train_and_infer.py --dataset open-llm-lb-v2 --routers mf,rorf,carrot-knn,carrot-knn-sbert
#   python train_and_infer.py --dataset routerbench --routers carrot-roberta --epochs 5

def autodetect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
### KNN ###
def tune_n_neighbors(X_train, Y_train, n_neighbors_range, metric='cosine', cv=5, task='classification'):

    best_score = -float('inf')
    best_n_neighbors = None

    for n_neighbors in n_neighbors_range:
        if task=='classification':
            KNN = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
            scores = cross_val_score(KNN, X_train, Y_train, cv=cv, scoring='roc_auc')  
        else:
            KNN = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
            scores = cross_val_score(KNN, X_train, Y_train, cv=cv, scoring='r2')
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_n_neighbors = n_neighbors

    return int(best_n_neighbors), best_score

def train_knn(E,
              Y,
              n_neighbors_range =[2**i for i in range(1,10)],
              metric='cosine',
              task='classification'):

    n_neighbors, cv_score = tune_n_neighbors(E, Y, metric = metric, n_neighbors_range = n_neighbors_range, task=task)       
    if task=='classification':
        KNN = KNeighborsClassifier(n_neighbors = int(n_neighbors), metric = metric)
    else:
        KNN = KNeighborsRegressor(n_neighbors = int(n_neighbors), metric = metric)
    KNN.fit(X=E, y=Y)
    return KNN
            
### RORF ###
def get_ND_label(y):
    if y[1]==1 and y[0]==0:
        return 0 
    elif y[1]==0 and y[0]==0:
        return 1 
    elif y[1]==1 and y[0]==1:
        return 2 
    elif y[1]==0 and y[0]==1:
        return 3 
    else:
        raise NotImplementedError

def get_rorf_labels(Y,small_model_ind,large_model_ind):
    return np.array([get_ND_label(y) for y in Y[:,[small_model_ind,large_model_ind]]])
    
def train_rorf(E, #embeddings
               Y,
               n_estimators = 100, #RF (https://github.com/Not-Diamond/RoRF/blob/main/run_trainer.sh)
               max_depth = 20, #RF (https://github.com/Not-Diamond/RoRF/blob/main/run_trainer.sh)
               random_state = RANDOM_STATE): 
    
    RF = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = RANDOM_STATE
        )
    RF = RandomForestClassifier(random_state=random_state)
    RF.fit(X=E, y=Y)
    return RF
            
### Matrix Factorization ###
def get_mf_labels(Y,small_model_ind,large_model_ind):
    return np.array([int(y[0]>=y[1]) for y in Y[:,[small_model_ind,large_model_ind]]])

class MFModel_Train(torch.nn.Module): #adapted from https://github.com/lm-sys/RouteLLM/blob/main/routellm/routers/matrix_factorization/train_matrix_factorization.py
    def __init__(
        self,
        embeddings,
        dim=128
    ):
        super().__init__()
        num_prompts,text_dim = embeddings.shape
        num_classes=1
        use_proj=True
        
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(1, dim)
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False) 
        self.Q.weight.data.copy_(embeddings)

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = nn.Linear(
            dim, num_classes, bias=False
        )  # bias should be False!

    def get_device(self):
        return self.P.weight.device

    def forward(self, prompt, test=False, alpha=0.05):
        prompt = prompt.to(self.get_device())
        model_embed = self.P(torch.tensor(0).to(self.get_device()))[None,:]
        model_embed = F.normalize(model_embed, p=2, dim=1)
        prompt_embed = self.Q(prompt)
        if not test:
            # adding noise to stablize the training
            prompt_embed += torch.randn_like(prompt_embed) * alpha
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(
            model_embed * prompt_embed
        ).squeeze()

    @torch.no_grad()
    def predict_proba(self, embedding):
        model_embed = self.P(torch.tensor(0).to(self.get_device()))[None,:]
        model_embed = F.normalize(model_embed, p=2, dim=1)
        prompt_embed = torch.tensor(embedding, dtype=torch.float32).to(self.get_device())

        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(
            model_embed * prompt_embed
        ).squeeze().detach().cpu().numpy()

def train_mf_model(E, #embeddings
                   Y,
                   epochs=100,
                   batch_size=64,
                   lr=3e-4,
                   weight_decay=1e-5,
                   val_split=VAL_SIZE,
                   validate_every=10,  # Validate every 'm' steps
                   random_state = RANDOM_STATE,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):

    torch.manual_seed(random_state)
    
    # Convert inputs to tensors if they aren't already
    embeddings = torch.tensor(E, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    
    # Create dataset and split into train/val
    dataset = TensorDataset(torch.arange(len(embeddings)), Y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and move to device
    model = MFModel_Train(embeddings).to(device)
    
    # Initialize optimizer and loss function
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Keep track of best validation accuracy and corresponding model weights
    best_val_acc = -float("inf")
    best_model_weights = None
    
    step = 0  # Global step counter
    for epoch in tqdm(range(epochs)):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for indices, labels in train_loader:
            indices, labels = indices.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(indices)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            # Increment step counter
            step += 1
            
            # Validation every 'validate_every' steps
            if step % validate_every == 0:
                model.eval()
                correct_val = 0
                total_val = 0

                val_probs_list = []
                val_labels_list = []
                with torch.no_grad():
                    for val_indices, val_labels in val_loader:
                        val_indices, val_labels = val_indices.to(device), val_labels.to(device)
                        
                        val_outputs = model(val_indices, test=True)  # Use test=True for validation
                        val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                        correct_val += (val_predicted == val_labels).sum().item()
                        total_val += val_labels.size(0)

                        val_probs_list += torch.sigmoid(val_outputs).cpu().numpy().tolist()
                        val_labels_list += val_labels.cpu().numpy().tolist()

                loglike = (np.array(val_labels_list)*np.log(np.array(val_probs_list))+(1-np.array(val_labels_list))*np.log(1-np.array(val_probs_list))).mean()
                
                val_acc = loglike # we use loglike instead of accuracy because accuracy does not provide enough fine-grained feedback (it ends up picking a classifier that classifies everything as one)
                #100 * correct_val / total_val
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_weights = model.state_dict().copy()
                
                #print(f'Step [{step}] - Val Acc: {val_acc:.2f}%')
                model.train()
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    #print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    
    return model
    
### Roberta ###
def prepare_datasets_prediction(test_texts,
                                tokenizer,
                                max_length=MAX_LENGTH):
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"]})
    return test_dataset
    
def prepare_datasets(train_texts,
                     val_texts,
                     train_labels,
                     val_labels,
                     tokenizer,
                     max_length=MAX_LENGTH):
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "labels": train_labels})
    val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "labels": val_labels})

    pad_token = tokenizer.pad_token_id
    print(f"Fraction of truncated training texts: {np.mean([t['input_ids'][-1]!=pad_token for t in train_dataset]):.2f}")
    print(f"Fraction of truncated validation texts: {np.mean([t['input_ids'][-1]!=pad_token for t in val_dataset]):.2f}")
    return train_dataset, val_dataset

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    mse = np.mean(np.square(labels-logits))
    return {"mse": mse}

def compute_metrics_for_classification(eval_pred, threshold=.5):
    sigmoid = torch.nn.Sigmoid()
    logits, labels = eval_pred
    probs = sigmoid(torch.Tensor(logits))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = np.zeros(labels.shape)
    y_true[np.where(labels >= threshold)] = 1
    return {'all_accuracy': (y_true == y_pred).mean()}

def get_compute_metrics(task):
    if task=='classification':
        return compute_metrics_for_classification
    else:
        return compute_metrics_for_regression

def train_roberta(X, #list of texts
                  Y, #response tensor
                  task, #'classification' vs 'regression'
                  output_dir = "../models/roberta/checkpoints",
                  model_name = MODEL_NAME,
                  learning_rate = 2e-5,
                  weight_decay = 0.01,
                  batch_size = 16,
                  gradient_accumulation_steps = 1,
                  early_stopping_patience = 50,
                  val_size = VAL_SIZE,
                  max_length = MAX_LENGTH,
                  warmup_steps = 1000,
                  eval_steps = 500,
                  max_steps = -1,  # HF: -1 means ignore, use num_train_epochs
                  num_train_epochs = 2,
                  random_state = RANDOM_STATE,
                  device = None):

    if device is None or device == 'auto':
        device = autodetect_device()

    assert task in ['classification', 'regression']

    # Loading model and tokenizer
    if Y.squeeze().shape==Y.shape: # assuming labels are at most bi-dimensional
        num_labels = Y.shape[1]
    else:
        num_labels = 1

    if task=='classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="regression", num_labels=num_labels)
    # Place model on target device explicitly. Using `device_map=device` in
    # from_pretrained disagrees with HF Trainer's device management and silently
    # falls back to CPU with extra transfers.
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare datasets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X, Y, test_size=val_size, random_state=random_state
    )
    train_dataset, val_dataset = prepare_datasets(
        train_texts, val_texts, train_labels, val_labels, tokenizer, max_length
    )

    # Generate training_args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=eval_steps,
        fp16=device.startswith("cuda"),
        report_to = "none",
        gradient_accumulation_steps=gradient_accumulation_steps,  
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=get_compute_metrics(task), 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])
    

    trainer.train()

    return trainer.model, tokenizer


def train_roberta_custom(X, Y, task,
                         model_name=MODEL_NAME,
                         learning_rate=2e-5,
                         weight_decay=0.01,
                         batch_size=8,
                         val_size=VAL_SIZE,
                         max_length=256,
                         warmup_ratio=0.1,
                         num_train_epochs=3,
                         grad_clip=1.0,
                         random_state=RANDOM_STATE,
                         device=None,
                         log_every=50):
    """Plain PyTorch training loop for RoBERTa. Sidesteps HF Trainer's MPS overhead
    that was producing 20s/step in real runs vs 2.6s/step in bare benchmarks.

    Defaults are tuned for 16 GB unified-memory Apple Silicon: batch=8, max_length=256.
    RoBERTa-base training with batch=16, seq=512 OOMs at ~20 GB on these machines."""
    if device is None or device == 'auto':
        device = autodetect_device()
    assert task in ['classification', 'regression']

    num_labels = Y.shape[1] if Y.squeeze().shape == Y.shape else 1

    if task == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, problem_type="multi_label_classification", num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, problem_type="regression", num_labels=num_labels)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X, Y, test_size=val_size, random_state=random_state)

    def tokenize_to_device(texts):
        enc = tokenizer(list(texts), truncation=True, padding=True,
                        max_length=max_length, return_tensors='pt')
        return {k: v.to(device) for k, v in enc.items()}

    print(f"  Tokenizing {len(train_texts)} train + {len(val_texts)} val texts...")
    train_enc = tokenize_to_device(train_texts)
    val_enc = tokenize_to_device(val_texts)
    train_labels_t = torch.tensor(np.asarray(train_labels), dtype=torch.float32).to(device)
    val_labels_t = torch.tensor(np.asarray(val_labels), dtype=torch.float32).to(device)
    if val_labels_t.ndim == 1:
        val_labels_t = val_labels_t.unsqueeze(1)
    if train_labels_t.ndim == 1:
        train_labels_t = train_labels_t.unsqueeze(1)

    n_train = train_labels_t.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = float('inf')
    best_state = None
    torch.manual_seed(random_state)

    step = 0
    t0 = time.time()
    for epoch in range(num_train_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, epoch_n = 0.0, 0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch_inputs = {k: v[idx] for k, v in train_enc.items()}
            batch_labels = train_labels_t[idx]

            optimizer.zero_grad(set_to_none=True)
            out = model(**batch_inputs, labels=batch_labels)
            loss = out.loss
            if not torch.isfinite(loss):
                print(f"  step {step}: non-finite loss, skipping")
                del out, loss
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            loss_val = loss.item()
            del out, loss

            epoch_loss += loss_val * idx.shape[0]
            epoch_n += idx.shape[0]
            step += 1
            if step % log_every == 0:
                elapsed = time.time() - t0
                eta = elapsed * (total_steps - step) / max(step, 1)
                print(f"  step {step}/{total_steps}  loss={loss_val:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"{elapsed / step:.2f}s/step  eta {eta / 60:.1f} min",
                      flush=True)
                if device == 'mps':
                    torch.mps.empty_cache()

        # Eval on val set
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for b in range(0, val_labels_t.shape[0], batch_size):
                batch_inputs = {k: v[b:b + batch_size] for k, v in val_enc.items()}
                batch_labels = val_labels_t[b:b + batch_size]
                out = model(**batch_inputs, labels=batch_labels)
                if torch.isfinite(out.loss):
                    val_loss += out.loss.item() * batch_labels.shape[0]
                    val_n += batch_labels.shape[0]
        val_loss /= max(val_n, 1)
        train_loss = epoch_loss / max(epoch_n, 1)
        print(f"  [epoch {epoch + 1}/{num_train_epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    new best val_loss, snapshotted")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model, tokenizer


def roberta_predict(model, tokenizer, texts, device, batch_size=32, max_length=MAX_LENGTH):
    """Batched inference. Returns raw logits for classification, raw outputs for regression."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            outs.append(logits.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


ALL_ROUTERS = ['mf', 'rorf', 'carrot-knn', 'carrot-knn-sbert', 'carrot-roberta',
               'carrot-roberta-perf', 'carrot-roberta-cost', 'roberta-binary']


def save_pred(preds_dir, key, value):
    np.save(os.path.join(preds_dir, f"{key}.npy"), value)


def load_or_none(preds_dir, key):
    path = os.path.join(preds_dir, f"{key}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['open-llm-lb-v2', 'routerbench', 'sprout'])
    parser.add_argument("--routers", type=str, default=','.join(ALL_ROUTERS),
                        help=f"Comma-separated subset of {ALL_ROUTERS}")
    parser.add_argument("--epochs", type=int, default=3,
                        help="num_train_epochs for RoBERTa")
    parser.add_argument("--trainer", type=str, default="custom",
                        choices=["hf", "custom"],
                        help="'custom' = plain PyTorch loop (fast on MPS); 'hf' = HuggingFace Trainer")
    args = parser.parse_args()

    dataset = args.dataset
    routers = [r.strip() for r in args.routers.split(',') if r.strip()]
    for r in routers:
        assert r in ALL_ROUTERS, f"Unknown router: {r}"
    epochs = args.epochs
    trainer_kind = args.trainer

    device = autodetect_device()
    print(f"Device: {device}")
    print(f"Dataset: {dataset}; routers: {routers}; epochs: {epochs}; trainer: {trainer_kind}")

    def fit_roberta(X, Y, task, output_dir):
        if trainer_kind == 'hf':
            return train_roberta(X, Y, task=task, output_dir=output_dir,
                                 num_train_epochs=epochs, device=device)
        return train_roberta_custom(X, Y, task=task, num_train_epochs=epochs, device=device)

    preds_dir = f"../data/{dataset}/preds"
    os.makedirs(preds_dir, exist_ok=True)

    SMALL_MODEL = LARGE_SMALL_MODELS[dataset]['SMALL_MODEL']
    LARGE_MODEL = LARGE_SMALL_MODELS[dataset]['LARGE_MODEL']

    ### Load data
    data = np.load(f"../data/{dataset}/{dataset}_data_train_test.npy", allow_pickle=True).item()
    small_model_ind = int(np.argmax(np.array(data['models']) == SMALL_MODEL))
    large_model_ind = int(np.argmax(np.array(data['models']) == LARGE_MODEL))

    for s in ['XOAI_train', 'XOAI_test', 'X_train', 'X_test', 'Y_train']:
        data[s] = np.array(data[s]).astype(float)
    if dataset == 'sprout':
        data['OT_train'] = np.array(data['OT_train']).astype(float)
    elif dataset == 'routerbench':
        data['C_train'] = np.array(data['C_train']).astype(float)

    ### Save shared meta once (test-set inputs, labels, etc.) — plotting reads from here
    meta_path = os.path.join(preds_dir, "meta.npy")
    if not os.path.exists(meta_path):
        meta = {
            'models': data['models'],
            'Y_test': data['Y_test'],
            'Q_test': data['Q_test'],
            'X_test': data['X_test'],
            'XOAI_test': data['XOAI_test'],
            'small_model_ind': small_model_ind,
            'large_model_ind': large_model_ind,
        }
        if 'C_test' in data:
            meta['C_test'] = data['C_test']
        if 'IT_test' in data:
            meta['IT_test'] = data['IT_test']
        if 'OT_test' in data:
            meta['OT_test'] = data['OT_test']
        np.save(meta_path, meta)
        print(f"Saved meta.npy with keys: {list(meta.keys())}")

    ### Train each requested router and save its per-router .npy
    if 'rorf' in routers:
        print("\n[rorf]")
        rorf = train_rorf(
            data['XOAI_train'],
            get_rorf_labels((data['Y_train'] > .5).astype(float), small_model_ind, large_model_ind)
        )
        save_pred(preds_dir, 'Y_hat_rorf', rorf.predict_proba(data['XOAI_test'])[:, -2:].sum(1))

    if 'mf' in routers:
        print("\n[mf]")
        mf = train_mf_model(
            data['XOAI_train'],
            get_mf_labels(data['Y_train'], small_model_ind, large_model_ind),
            device=device,
        )
        # sigmoid the logits so route_pairwise's alpha sweep over [0,1] reaches both endpoints
        save_pred(preds_dir, 'Y_hat_mf', sigmoid(mf.predict_proba(data['XOAI_test'])))

    if 'carrot-knn' in routers:
        print("\n[carrot-knn]")
        knn = train_knn(data['XOAI_train'], data['Y_train'], task='regression')
        save_pred(preds_dir, 'Y_hat_carrot-knn', knn.predict(data['XOAI_test']))

        if dataset == 'sprout':
            ot_knn = train_knn(data['XOAI_train'], data['OT_train'], task='regression')
            save_pred(preds_dir, 'OT_hat_carrot-knn', ot_knn.predict(data['XOAI_test']))
        elif dataset == 'routerbench':
            mu = np.mean(data['C_train'], axis=0, keepdims=True)
            std = np.std(data['C_train'], axis=0, keepdims=True)
            Z = (data['C_train'] - mu) / std
            c_knn = train_knn(data['XOAI_train'], Z, task='regression')
            save_pred(preds_dir, 'C_hat_carrot-knn', mu + std * c_knn.predict(data['XOAI_test']))

    if 'carrot-knn-sbert' in routers:
        print("\n[carrot-knn-sbert]")
        knn_sbert = train_knn(data['X_train'], data['Y_train'], task='regression')
        save_pred(preds_dir, 'Y_hat_carrot-knn-sbert', knn_sbert.predict(data['X_test']))

        if dataset == 'sprout':
            ot_knn = train_knn(data['X_train'], data['OT_train'], task='regression')
            save_pred(preds_dir, 'OT_hat_carrot-knn-sbert', ot_knn.predict(data['X_test']))
        elif dataset == 'routerbench':
            mu = np.mean(data['C_train'], axis=0, keepdims=True)
            std = np.std(data['C_train'], axis=0, keepdims=True)
            Z = (data['C_train'] - mu) / std
            c_knn = train_knn(data['X_train'], Z, task='regression')
            save_pred(preds_dir, 'C_hat_carrot-knn-sbert', mu + std * c_knn.predict(data['X_test']))

    # Treat 'carrot-roberta' as shorthand for running both perf and cost heads.
    run_perf = ('carrot-roberta' in routers) or ('carrot-roberta-perf' in routers)
    run_cost = ('carrot-roberta' in routers) or ('carrot-roberta-cost' in routers)

    if run_perf:
        print("\n[carrot-roberta perf]")
        model, tokenizer = fit_roberta(
            list(data['Q_train']), data['Y_train'], task='classification',
            output_dir=f"../models/{dataset}/perf/roberta/checkpoints",
        )
        y_logits = roberta_predict(model, tokenizer, list(data['Q_test']), device)
        save_pred(preds_dir, 'Y_hat_carrot-roberta', sigmoid(y_logits))
        del model

    if run_cost:
        if dataset == 'sprout':
            print("\n[carrot-roberta cost (output tokens)]")
            model, tokenizer = fit_roberta(
                list(data['Q_train']), data['OT_train'], task='regression',
                output_dir=f"../models/{dataset}/cost/roberta/checkpoints",
            )
            ot_hat = roberta_predict(model, tokenizer, list(data['Q_test']), device)
            save_pred(preds_dir, 'OT_hat_carrot-roberta', ot_hat)
            del model
        elif dataset == 'routerbench':
            print("\n[carrot-roberta cost (standardized)]")
            mu = np.mean(data['C_train'], axis=0, keepdims=True)
            std = np.std(data['C_train'], axis=0, keepdims=True)
            Z = (data['C_train'] - mu) / std
            model, tokenizer = fit_roberta(
                list(data['Q_train']), Z, task='regression',
                output_dir=f"../models/{dataset}/cost/roberta/checkpoints",
            )
            z_hat = roberta_predict(model, tokenizer, list(data['Q_test']), device)
            save_pred(preds_dir, 'C_hat_carrot-roberta', mu + std * z_hat)
            del model
        # No cost head for open-llm-lb-v2 (costs are deterministic from input tokens).

    if 'roberta-binary' in routers:
        print("\n[roberta-binary]")
        # Same binary label as MF/RoRF: y=1 if small model >= large model on this prompt.
        Y_binary = get_mf_labels(
            data['Y_train'], small_model_ind, large_model_ind
        ).reshape(-1, 1).astype(float)
        model, tokenizer = fit_roberta(
            list(data['Q_train']), Y_binary, task='classification',
            output_dir=f"../models/{dataset}/binary/roberta/checkpoints",
        )
        y_logits = roberta_predict(model, tokenizer, list(data['Q_test']), device)
        # Single-column output; sigmoid to match other binary routers' [0,1] range.
        save_pred(preds_dir, 'Y_hat_roberta-binary', sigmoid(y_logits).squeeze())
        del model

    print(f"\nDone. Predictions written under {preds_dir}/")