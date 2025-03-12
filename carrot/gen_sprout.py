from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
from data_utils import *

models = ['aws-claude-3-5-sonnet-v1', 'aws-titan-text-premier-v1', 'openai-gpt-4o', 'openai-gpt-4o-mini', 'wxai-granite-3-2b-instruct-8k-max-tokens', 'wxai-granite-3-8b-instruct-8k-max-tokens', 'wxai-llama-3-1-70b-instruct', 'wxai-llama-3-1-8b-instruct', 'wxai-llama-3-2-1b-instruct', 'wxai-llama-3-2-3b-instruct', 'wxai-llama-3-3-70b-instruct', 'wxai-llama-3-405b-instruct', 'wxai-mixtral-8x7b-instruct-v01']
sprout = load_dataset("CARROT-LLM-Routing/SPROUT")

data = {'Q_train':sprout['train']['prompt']+sprout['validation']['prompt'],
        'Q_test':sprout['test']['prompt'],
        'Y_train':np.vstack((np.array([[d['score'] for d in sprout['train'][m]] for m in models]).T,np.array([[d['score'] for d in sprout['validation'][m]] for m in models]).T)),
        'Y_test':np.array([[d['score'] for d in sprout['test'][m]] for m in models]).T,
        'IT_train':np.vstack((np.array([[d['num_input_tokens'] for d in sprout['train'][m]] for m in models]).T,np.array([[d['num_input_tokens'] for d in sprout['validation'][m]] for m in models]).T)),
        'IT_test':np.array([[d['num_input_tokens'] for d in sprout['test'][m]] for m in models]).T,
        'OT_train':np.vstack((np.array([[d['num_output_tokens'] for d in sprout['train'][m]] for m in models]).T,np.array([[d['num_output_tokens'] for d in sprout['validation'][m]] for m in models]).T)),
        'OT_test':np.array([[d['num_output_tokens'] for d in sprout['test'][m]] for m in models]).T,
        'models':models}

data['X_train'] = embedding_model.encode(data['Q_train']).tolist()
data['X_test'] = embedding_model.encode(data['Q_test']).tolist()

data['Q_test'] = [truncate_text(s) for s in data['Q_test']]
data['Q_train'] = [truncate_text(s) for s in data['Q_train']]
data['XOAI_test'] = get_oai_embs(data['Q_test'])
data['XOAI_train'] = get_oai_embs(data['Q_train'])

np.save("../data/sprout/sprout_data_train_test.npy",data)