import json
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from utils import *
from constants import *

bench = BENCH
test_size = TEST_SIZE
random_state = RANDOM_STATE

with open('data/data_QA_cost_embedding.json', "r") as datafile:
    data_QA = json.load(datafile)

data_Y = np.load("data/new_leaderboard_processed_20241205.pickle", allow_pickle=True)
M = [data_Y[k]['models'] for k in BENCHMARKS[bench]]
M = np.sort(list(reduce(set.intersection, map(set, M)))).tolist()
Y = [data_Y[k]['correctness'][[int(np.argmax(np.array(data_Y[k]['models'])==m)) for m in M]] for k in BENCHMARKS[bench]]
Y = np.hstack(Y)
data_Y[bench] = {}
data_Y[bench]['correctness'] = Y.T
data_Y[bench]['models'] = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in M]

data_QA[bench] = {}
data_QA[bench]['Es'] = np.vstack([np.array(data_QA[k]['Es']) for k in BENCHMARKS[bench]])
data_QA[bench]['Es_OAI'] = np.vstack([np.array(data_QA[k]['Es_OAI']) for k in BENCHMARKS[bench]])
data_QA[bench]['Qs'] = flatten([data_QA[k]['Qs'] for k in BENCHMARKS[bench]])
data_QA[bench]['input_cost'] = np.vstack([np.array([data_QA[k]['input_cost'][model] for model in data_Y[bench]['models']]).T for k in BENCHMARKS[bench]])
data_QA[bench]['input_cost'] = data_QA[bench]['input_cost']#normalizing to be in [0,1]

Q_train, Q_test, X_train, X_test, XOAI_train, XOAI_test, Y_train, Y_test, C_train, C_test= train_test_split(data_QA[bench]['Qs'],
                                                                                                            data_QA[bench]['Es'],
                                                                                                            data_QA[bench]['Es_OAI'],
                                                                                                            data_Y[bench]['correctness'],
                                                                                                            data_QA[bench]['input_cost'],
                                                                                                            test_size=test_size,
                                                                                                            random_state=random_state)

data = {'Q_train':Q_train, 'Q_test':Q_test,
        'X_train':X_train, 'X_test':X_test,
        'XOAI_train':XOAI_train, 'XOAI_test':XOAI_test,
        'Y_train':Y_train, 'Y_test':Y_test,
        'C_train':C_train, 'C_test':C_test,
        'models': data_Y[bench]['models']}

np.save("data/data_train_test.npy",data)