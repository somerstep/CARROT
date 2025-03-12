from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import json
from functools import reduce
from sklearn.model_selection import train_test_split
from data_utils import *
from constants import *

BENCH = 'all'
MODELS = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in OPEN_MODELS]
BENCHMARKS = OPEN_BENCHMARKS

tokenizer_dict = {}
for model in MODELS:
    tokenizer_dict[model] = AutoTokenizer.from_pretrained(model, use_fast=False)
    
with open('../data/open-llm-lb-v2/data_QA.json', "r") as datafile:
    data_QA = json.load(datafile)

for sce in tqdm(data_QA.keys()):
    print(f"************** {sce} **************")
    data_QA[sce]['tokens'] = np.array([[len(tokenizer_dict[model](q)['input_ids']) for q in data_QA[sce]['Ps']] for model in MODELS]).T
    data_QA[sce]['X'] = embedding_model.encode(data_QA[sce]['Qs']).tolist()
    data_QA[sce]['XOAI'] = get_oai_embs(data_QA[sce]['Qs'])

bench = BENCH
data_Y = np.load("../data/open-llm-lb-v2/new_leaderboard_processed_20241205.pickle", allow_pickle=True)
M = [data_Y[k]['models'] for k in BENCHMARKS[bench]]
M = np.sort(list(reduce(set.intersection, map(set, M)))).tolist()
Y = [data_Y[k]['correctness'][[int(np.argmax(np.array(data_Y[k]['models'])==m)) for m in M]] for k in BENCHMARKS[bench]]
Y = np.hstack(Y)
data_Y[bench] = {}
data_Y[bench]['correctness'] = Y.T
data_Y[bench]['models'] = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in M]

data_QA[bench] = {}
data_QA[bench]['X'] = np.vstack([np.array(data_QA[k]['X']) for k in BENCHMARKS[bench]])
data_QA[bench]['XOAI'] = np.vstack([np.array(data_QA[k]['XOAI']) for k in BENCHMARKS[bench]])
data_QA[bench]['Qs'] = flatten([data_QA[k]['Qs'] for k in BENCHMARKS[bench]])
data_QA[bench]['tokens'] = np.vstack([data_QA[k]['tokens'] for k in BENCHMARKS[bench]])

Q_train, Q_test, X_train, X_test, XOAI_train, XOAI_test, Y_train, Y_test, IT_train, IT_test= train_test_split(data_QA[bench]['Qs'],
                                                                                                              data_QA[bench]['X'],
                                                                                                              data_QA[bench]['XOAI'],
                                                                                                              data_Y[bench]['correctness'],
                                                                                                              data_QA[bench]['tokens'],
                                                                                                              test_size=TEST_SIZE,
                                                                                                              random_state=RANDOM_STATE)

data = {'Q_train':Q_train, 'Q_test':Q_test,
        'X_train':X_train, 'X_test':X_test,
        'XOAI_train':XOAI_train, 'XOAI_test':XOAI_test,
        'Y_train':Y_train, 'Y_test':Y_test,
        'IT_train':IT_train, 'IT_test':IT_test,
        'models': data_Y[bench]['models']}

np.save("../data/open-llm-lb-v2/open-llm-lb-v2_data_train_test.npy",data)