import json
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.model_selection import KFold
from constants import *
from utils import *
    
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
embedding_model_oai = "text-embedding-3-small"
openai_client = OpenAI(api_key=OPENAI_API_KEY)
max_batch_size = 2000 #needed for openai api (they do not accept large batches)

MODELS = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in MODELS]

with open('data/data_QA.json', "r") as datafile:
    data_QA = json.load(datafile)

tokenizer_dict = {}
for model in MODELS:
    tokenizer_dict[model] = AutoTokenizer.from_pretrained(model)

for sce in tqdm(data_QA.keys()):
    print(f"************** {sce} **************")
    data_QA[sce]['input_cost'] = {}
    tokens = [tokenizer_dict[model](q)['input_ids'] for q in data_QA[sce]['Ps']]
    tokens = np.array([len(t) for t in tokens])
    for model in MODELS:
        data_QA[sce]['input_cost'][model] = ((COSTS[model]/1e6)*tokens).tolist()

    inputs = data_QA[sce]['Qs']
    data_QA[sce]['Es'] = embedding_model.encode(inputs).tolist()

    batch_size = min(len(inputs)/2,max_batch_size)
    kf = KFold(n_splits=int(len(inputs)//batch_size+1))
    response = []
    for i, (_, index) in enumerate(kf.split(inputs)):
        response.append(openai_client.embeddings.create(
                model=embedding_model_oai, input=np.array(inputs)[index].tolist(), encoding_format="float"
        ))
        response[-1] = [r.embedding for r in response[-1].data]
    response = flatten(response)
    
    data_QA[sce]['Es_OAI'] = response

with open('data/data_QA_cost_embedding.json', "w") as datafile:
    json.dump(data_QA, datafile, indent=4)