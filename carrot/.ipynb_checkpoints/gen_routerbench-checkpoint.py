from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from data_utils import *
from constants import *

df = pd.read_pickle('../data/routerbench/routerbench_0shot.pkl')
df = df[[not s.lower().startswith('chinese') for s in df['eval_name'].values]] # remove the chinese questions
df.prompt = [s[2:-2] for s in df['prompt']]
models = [str(s) for s in df.columns.values[3:14]]

#labels
Y = (df.values[:,3:14]).astype('float')
cost_columns = [model+'|total_cost' for model in models]
C = np.array(df[cost_columns].values, dtype = 'float32')

#embs
Q = list(df.prompt)
X = embedding_model.encode(Q)

Q2 = [truncate_text(s) for s in Q]
XOAI = []
batch_size = 1000  
for i in tqdm(range(0, len(Q2), batch_size)):
    XOAI += get_oai_embs(Q2[i:i + batch_size])

# splitting and saving
Q_train, Q_test, X_train, X_test, XOAI_train, XOAI_test, Y_train, Y_test, C_train, C_test= train_test_split(Q,
                                                                                                            X,
                                                                                                            XOAI,
                                                                                                            Y,
                                                                                                            C,
                                                                                                            test_size=TEST_SIZE,
                                                                                                            random_state=RANDOM_STATE)

data = {'Q_train':Q_train, 'Q_test':Q_test,
        'X_train':X_train, 'X_test':X_test,
        'XOAI_train':XOAI_train, 'XOAI_test':XOAI_test,
        'Y_train':Y_train, 'Y_test':Y_test,
        'C_train':C_train, 'C_test':C_test,
        'models': models}

np.save("../data/routerbench/routerbench_data_train_test.npy",data)