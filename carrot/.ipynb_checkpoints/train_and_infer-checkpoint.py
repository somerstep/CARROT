import numpy as np
import argparse
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback, get_scheduler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss
from constants import *

# python train_and_infer.py --dataset 'open-llm-lb-v2' OR 'sprout' OR 'routerbench' (not implemented yet)

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
                  eval_steps = 200,
                  max_steps = 500000,
                  random_state = RANDOM_STATE,
                  device = 'cuda'):

    assert task in ['classification', 'regression']

    # Loading model and tokenizer
    if Y.squeeze().shape==Y.shape: # assuming labels are at most bi-dimensional
        num_labels = Y.shape[1]
    else:
        num_labels = 1

    if task=='classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=num_labels, device_map = device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="regression", num_labels=num_labels, device_map = device)

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
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=9999, #
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=eval_steps,
        fp16=True if device.startswith("cuda") else False,
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

    return trainer

if __name__=="__main__":

    ### Inputs
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    dataset = args.dataset

    assert dataset in ['open-llm-lb-v2','routerbench','sprout']

    
    SMALL_MODEL, LARGE_MODEL = LARGE_SMALL_MODELS[dataset]['SMALL_MODEL'], LARGE_SMALL_MODELS[dataset]['LARGE_MODEL']
    
    ### Load data
    data = np.load(f"../data/{dataset}/{dataset}_data_train_test.npy", allow_pickle=True).item()
    small_model_ind = int(np.argmax(np.array(data['models'])==SMALL_MODEL))
    large_model_ind = int(np.argmax(np.array(data['models'])==LARGE_MODEL))
    
    ### Train and infer
    for s in ['XOAI_train','XOAI_test','X_train','X_test','Y_train']:
        data[s] = np.array(data[s]).astype(float)
    if dataset == 'sprout':
        data['OT_train'] = np.array(data['OT_train']).astype(float)
    elif dataset == 'routerbench':
        data['C_train'] = np.array(data['C_train']).astype(float)

    rorf = train_rorf(data['XOAI_train'], get_rorf_labels((data['Y_train']>.5).astype(float),small_model_ind,large_model_ind))
    data['Y_hat_rorf'] = rorf.predict_proba(data['XOAI_test'])[:,-2:].sum(1)
    
    mf = train_mf_model(data['XOAI_train'], get_mf_labels(data['Y_train'],small_model_ind,large_model_ind))
    data['Y_hat_mf'] = mf.predict_proba(data['XOAI_test'])

    knn = train_knn(data['XOAI_train'], data['Y_train'], task='regression')
    data['Y_hat_carrot-knn'] = knn.predict(data['XOAI_test'])
    
    knn_sbert = train_knn(data['X_train'], data['Y_train'], task='regression')
    data['Y_hat_carrot-knn-sbert'] = knn_sbert.predict(data['X_test'])

    roberta = train_roberta(data['Q_train'], data['Y_train'], task='classification', output_dir = f"../models/{dataset}/perf/roberta/checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data['Y_hat_carrot-roberta'] = sigmoid(roberta.predict(prepare_datasets_prediction(data['Q_test'], tokenizer)).predictions)

    if dataset == 'sprout':
        knn = train_knn(data['XOAI_train'], data['OT_train'], task='regression')
        data['OT_hat_carrot-knn'] = knn.predict(data['XOAI_test'])
        
        knn_sbert = train_knn(data['X_train'], data['OT_train'], task='regression')
        data['OT_hat_carrot-knn-sbert'] = knn_sbert.predict(data['X_test'])
    
        roberta = train_roberta(data['Q_train'], data['OT_train'], task='regression', output_dir = f"../models/{dataset}/cost/roberta/checkpoints")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        data['OT_hat_carrot-roberta'] = roberta.predict(prepare_datasets_prediction(data['Q_test'], tokenizer)).predictions
        
    elif dataset == 'routerbench':
        mu = np.mean(data['C_train'], axis=0, keepdims=True)
        std = np.std(data['C_train'], axis=0, keepdims=True)
        Z = (data['C_train'] - mu)/std
        
        knn = train_knn(data['XOAI_train'], Z, task='regression')
        Z_hat = knn.predict(data['XOAI_test'])
        data['C_hat_carrot-knn'] = mu + std*Z_hat
        
        knn_sbert = train_knn(data['X_train'], Z, task='regression')
        Z_hat = knn_sbert.predict(data['X_test'])
        data['C_hat_carrot-knn-sbert'] = mu + std*Z_hat
    
        roberta = train_roberta(data['Q_train'], Z, task='regression', output_dir = f"../models/{dataset}/cost/roberta/checkpoints")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        Z_hat = roberta.predict(prepare_datasets_prediction(data['Q_test'], tokenizer)).predictions
        data['C_hat_carrot-roberta'] = mu + std*Z_hat

    np.save(f"../data/{dataset}/{dataset}_data_train_test_preds.npy", data)