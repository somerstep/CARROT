import torch
from transformers import AutoTokenizer
torch._dynamo.config.suppress_errors = True

from safetensors.torch import load_file
import numpy as np
from joblib import load
from tqdm.auto import tqdm
from constants import *
from train_models import *


device = torch.device("cuda:1")

def sigmoid(z):
    return 1/(1+np.exp(-z))

def prepare_datasets_prediction(train_texts, val_texts, tokenizer, max_length):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"]})
    val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"]})

    pad_token = tokenizer.pad_token_id
    print(f"Fraction of truncated training texts: {np.mean([t['input_ids'][-1]!=pad_token for t in train_dataset]):.2f}")
    print(f"Fraction of truncated validation texts: {np.mean([t['input_ids'][-1]!=pad_token for t in val_dataset]):.2f}")
    return train_dataset, val_dataset
    
def get_bert_pred(Q_test, model_path):

    max_length = MAX_LENGTH
    batch_size = BATCH_SIZE
    #model_path = "./models/bert/final_model/best_model"  
    
    # Example input data (replace `test_texts` with your actual data)
    test_texts = Q_test
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer
    )
    
    test_dataset, _ = prepare_datasets_prediction(Q_test, Q_test, tokenizer, max_length)
    
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    
    return sigmoid(logits)

if __name__=='__main__':

    y_hat = {}
    
    ##### Data ######
    data = np.load("data/data_train_test.npy", allow_pickle=True).item()
    Q_test = data['Q_test']
    X_test = data['X_test']
    Y_test = data['Y_test']
    XOAI_test = data['XOAI_test']
    num_labels = data['Y_test'].shape[1]
    
    ##### Mat Fact - RouteLLM ######
    X_train = data['X_train']
    XOAI_train = data['XOAI_train']
    for model in ['mf_routellm','mf_OAI_routellm']:
        file_path = f'./models/MF_routellm/{model}.pht'
        MF = MFModel_Train(torch.tensor({'mf_routellm':X_train,'mf_OAI_routellm':XOAI_train}[model]))
        MF.load_state_dict(torch.load(file_path))
        MF.eval()
        y_hat[model] = sigmoid(MF.predict_proba({'mf_routellm':X_test,'mf_OAI_routellm':XOAI_test}[model]))
    
    ##### KNN - ours ######
    for model in ['knn_ours','knn_OAI_ours']:
        file_path = f'./models/KNN_ours/{model}.joblib'
        KNN = load(file_path)
        predictions = KNN.predict_proba({'knn_ours':X_test,'knn_OAI_ours':XOAI_test}[model])
        predictions = np.array([[y[1] for y in yy] for yy in predictions]).T
        y_hat[model] = predictions

    ##### KNN - lamb ######
    for i in range(len(LAMB)):
        for model in ['knn_OAI']: #'knn',
            file_path = f'./models/KNN_lamb/{model}_lamb-{i}.joblib'
            KNN = load(file_path)
            if model == 'knn':
                predictions = KNN.predict(X_test)
            else:
                predictions = KNN.predict(XOAI_test)
            y_hat[f'{model}_lamb-{i}'] = predictions
        
    ##### RORF ######
    for model in ['rorf_nd','rorf_OAI_nd']:
        file_path = f'./models/RORF/{model}.joblib'
        RF = load(file_path)
        predictions = RF.predict_proba({'rorf_nd':X_test,'rorf_OAI_nd':XOAI_test}[model])
        y_hat[model] = predictions[:,-2:].sum(1)
        
    ##### Bert - Ours ######  
    y_hat['bert_ours'] = get_bert_pred(Q_test, "./models/bert_ours/final_model/best_model")

    ##### Bert - routellm ######  
    y_hat['bert_routellm'] = get_bert_pred(Q_test, "./models/bert_routellm/final_model/best_model")

    ##### Saving ######
    np.save("data/predictions.npy", y_hat)