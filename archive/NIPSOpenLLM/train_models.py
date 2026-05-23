import numpy as np
import torch
import wandb
import random
from joblib import dump
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset
from constants import *
import os

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

wandb_project = "routing2" 
os.environ["WANDB_PROJECT"]= wandb_project
os.environ["WANDB_API_KEY"] = ## ADD THIS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

##### KNN #####
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

##### RF #####
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

def tune_random_forest(X_train, Y_train, min_samples_leaf_range, max_features_range, cv=5):
    best_score = 0
    best_params = {}

    for min_samples_leaf in min_samples_leaf_range:
        for max_features in max_features_range:
            rf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,  # Fixed number of estimators
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=RANDOM_STATE
            )
            scores = cross_val_score(rf, X_train, Y_train, cv=cv, scoring='roc_auc')  # Change scoring if needed
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'min_samples_leaf': min_samples_leaf, 'max_features': max_features}

    return best_params, best_score

##### MF #####
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

def train_mf_model(embeddings,
                   Y,
                   epochs=5,
                   batch_size=64,
                   lr=3e-4,
                   weight_decay=1e-5,
                   val_split=0.2,
                   validate_every=10,  # Validate every 'm' steps
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    # Convert inputs to tensors if they aren't already
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
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
    best_val_acc = 0.0
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
                
                with torch.no_grad():
                    for val_indices, val_labels in val_loader:
                        val_indices, val_labels = val_indices.to(device), val_labels.to(device)
                        
                        val_outputs = model(val_indices, test=True)  # Use test=True for validation
                        val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                        correct_val += (val_predicted == val_labels).sum().item()
                        total_val += val_labels.size(0)
                
                val_acc = 100 * correct_val / total_val
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_weights = model.state_dict().copy()
                
                #print(f'Step [{step}] - Val Acc: {val_acc:.2f}%')
                model.train()
        
        # Calculate epoch metrics
        train_acc = 100 * correct_train / total_train
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    #print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    
    return model.to('cpu')

##### BERT #####
# Create Dataset objects
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Define evaluation metric
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    logits = torch.sigmoid(torch.tensor(logits)).numpy()  # Apply sigmoid to logits
    auc = roc_auc_score(labels, logits)
    return {"roc_auc": auc}

# Function to prepare datasets
def prepare_datasets(train_texts, val_texts, train_labels, val_labels, tokenizer, max_length):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "labels": train_labels})
    val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "labels": val_labels})

    pad_token = tokenizer.pad_token_id
    print(f"Fraction of truncated training texts: {np.mean([t['input_ids'][-1]!=pad_token for t in train_dataset]):.2f}")
    print(f"Fraction of truncated validation texts: {np.mean([t['input_ids'][-1]!=pad_token for t in val_dataset]):.2f}")
    return train_dataset, val_dataset

    
# Function to create a Trainer
def create_trainer(model, train_dataset, val_dataset, tokenizer, num_epochs, learning_rate, weight_decay, batch_size, gradient_accumulation_steps, warmup_steps, eval_steps, max_steps, output_dir, run_name, is_final=False):
    # Generate a meaningful run name
    training_args = TrainingArguments(
        output_dir=output_dir + "/" + run_name,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True if device.startswith("cuda") else False,
        report_to=["wandb"],  # Log to wandb only for the final model
        run_name=run_name,  # Set the wandb run name
    
        # Gradient accumulation
        gradient_accumulation_steps=gradient_accumulation_steps,  # Define this variable elsewhere, e.g., 8
    
        # Scheduler with warmup
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,  # Define this variable elsewhere, e.g., 500
    )


    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Replace with your metric function
    )


# Function to initialize a model
def initialize_model(num_labels):
    return AutoModelForSequenceClassification.from_pretrained(BERT_NAME, num_labels=num_labels)

def tune_train_bert(Q_train, Y_train, method,
                    learning_rates = [5e-5, 1e-5],
                    weight_decays = [1e-2, 1e-4],
                    gradient_accumulation_steps = [1, 2, 4, 8]):
    
    # Define parameters
    batch_size = BATCH_SIZE
    warmup_steps = 500
    eval_steps = 300
    max_steps_val = 3000
    max_steps = 10000
    val_size = 0.2
    random_state = RANDOM_STATE
    max_length = MAX_LENGTH
    num_labels = Y_train.shape[1]
    num_epochs = 9999  # Use max_steps instead

    # Prepare datasets
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        Q_train, Y_train, test_size=val_size, random_state=random_state
    )
    train_dataset, val_dataset = prepare_datasets(
        train_texts, val_texts, train_labels, val_labels, tokenizer, max_length
    )

    # Hyperparameter tuning
    #if len(learning_rates)==1 and len(weight_decays)==1 and len(gradient_accumulation_steps)==1:
    #else:
    best_metric = -np.inf
    best_params = {}

    for lr, wd, ac in itertools.product(learning_rates, weight_decays, gradient_accumulation_steps):
        run_name = f"method-{method}_lr={lr}_wd={wd}_ac={ac}"
        print(f"Training/validating with learning_rate={lr}, weight_decay={wd}, grad_acum={ac}")
        wandb.init(
            project=wandb_project,
            config={"learning_rate": lr, "weight_decay": wd, "gradient_accumulation_steps": ac},
            name=run_name
        )

        model = initialize_model(num_labels)
        trainer = create_trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            num_epochs=num_epochs,
            learning_rate=lr,
            weight_decay=wd,
            batch_size=batch_size,
            gradient_accumulation_steps=ac,
            warmup_steps=warmup_steps,
            eval_steps=eval_steps,
            max_steps=max_steps_val,
            output_dir=f"./models/bert_{method}",
            run_name=run_name
        )
        trainer.train()
        metrics = trainer.evaluate()
        wandb.finish()

        roc_auc = metrics.get("eval_roc_auc", -np.inf)
        print(f"ROC AUC for learning_rate={lr}, weight_decay={wd}: {roc_auc}, gradient_accumulation_steps={ac}")
        if roc_auc > best_metric:
            best_metric = roc_auc
            best_params = {
                "learning_rate": lr,
                "weight_decay": wd,
                "gradient_accumulation_steps": ac
            }

    print("Best hyperparameters:", best_params)
    print("Best ROC AUC:", best_metric)

    # Train final model with the best hyperparameters
    print("Training final model with the best hyperparameters...")
    run_name = f"final_lr={best_params['learning_rate']}_wd={best_params['weight_decay']}_ac={best_params['gradient_accumulation_steps']}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        config=best_params
    )

    final_model = initialize_model(num_labels)
    final_trainer = create_trainer(
        model=final_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        batch_size=batch_size,
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        max_steps=max_steps,
        output_dir=f"./models/bert_{method}/final_model",
        run_name=run_name,
        is_final=True,
    )
    final_trainer.train()
    wandb.finish()

    final_trainer.save_model(f"./models/bert_{method}/final_model/best_model")

if __name__ == "__main__":

    ##### Data ######
    data = np.load("data/data_train_test.npy", allow_pickle=True).item()
    Q_train = data['Q_train']
    C_train = data['C_train']
    X_train = data['X_train']
    XOAI_train = data['XOAI_train']
    Y_train = data['Y_train']

    ##### Mat Fact ######
    large_model_ind = int(np.argmax(np.array(data['models'])==LARGE_MODEL))
    small_model_ind = int(np.argmax(np.array(data['models'])==SMALL_MODEL))

    YMF_train = np.array([int(y[0]>=y[1]) for y in data['Y_train'][:,[small_model_ind,large_model_ind]]])

    for x_name in ['','_OAI']:

        file_path = f'./models/MF_routellm/mf{x_name}_routellm.joblib'
        if 'OAI' in x_name: X = XOAI_train
        else: X = X_train

        if not os.path.exists(file_path):
            torch.save(train_mf_model(X, YMF_train).state_dict(), file_path)
    
    ##### KNN - ours ######
    n_neighbors_range =[2**i for i in range(1,10)]

    for x_name in ['','_OAI']:

        file_path = f'./models/KNN_ours/knn{x_name}_ours.joblib'
        if 'OAI' in x_name: X = XOAI_train
        else: X = X_train

        if not os.path.exists(file_path):
            n_neighbors, cv_score = tune_n_neighbors(X_train, Y_train, n_neighbors_range = n_neighbors_range)       
            KNN = KNeighborsClassifier(n_neighbors = int(n_neighbors), metric = 'cosine')
            KNN.fit(X=X, y=Y_train)
            dump(KNN, file_path)


    ##### KNN - lamb ######
    lamb = np.array(LAMB)[None,None,:]
    Y_lamb = (1-lamb)*Y_train[:,:,None] - lamb*C_train[:,:,None]

    for x_name in ['_OAI']:
    
        if 'OAI' in x_name: X = XOAI_train
        else: X = X_train
            
        for i,lamb in tqdm(enumerate(LAMB)):
            file_path = f'./models/KNN_lamb/knn{x_name}_lamb-{i}.joblib'
            if not os.path.exists(file_path):
                n_neighbors, cv_score = tune_n_neighbors(X_train, Y_train, n_neighbors_range = n_neighbors_range, task='regression')      
                KNN = KNeighborsRegressor(n_neighbors = n_neighbors, metric = 'cosine')
                KNN.fit(X=X, y=Y_lamb[:,:,i])
                dump(KNN, file_path)

    ##### RF - Non-Diamond ##### https://www.notdiamond.ai/blog/rorf
    YND_train = np.array([get_ND_label(y) for y in data['Y_train'][:,[small_model_ind,large_model_ind]]])

    for x_name in ['','_OAI']:

        file_path = f'./models/RORF/rorf{x_name}_nd.joblib'
        if 'OAI' in x_name: X = XOAI_train
        else: X = X_train

        if not os.path.exists(file_path):
            RF = RandomForestClassifier(
                    n_estimators = N_ESTIMATORS,
                    max_depth = MAX_DEPTH,
                    random_state = RANDOM_STATE
                )
            RF = RandomForestClassifier(random_state=RANDOM_STATE)
            RF.fit(X=X, y=YND_train)
            dump(RF, file_path)

    ##### BERTs #####
    for method in ['routellm', 'ours']:
        file_path = f'./models/bert_{method}/final_model/best_model/model.safetensors'

        if not os.path.exists(file_path):
            if method=='routellm':
                Y = YMF_train.reshape(-1,1).astype(float)
            else:
                Y = Y_train
            tune_train_bert(Q_train, Y, method) 
