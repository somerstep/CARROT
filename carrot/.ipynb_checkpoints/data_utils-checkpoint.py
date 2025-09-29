from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from datasets import load_dataset
from openai import OpenAI
import numpy as np
import tiktoken

OPENAI_API_KEY = "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"
embedding_model_oai = "text-embedding-3-small"
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

def flatten(xss):
    return [x for xs in xss for x in xs]

def truncate_text(text, model="text-embedding-3-small", max_tokens=8192):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
    
def get_oai_embs(inputs, max_batch_size = 2000): #batching needed for openai api (they do not accept large batches)
    batch_size = min(len(inputs)/2,max_batch_size)
    kf = KFold(n_splits=int(len(inputs)//batch_size+1))
    response = []
    for i, (_, index) in enumerate(kf.split(inputs)):
        response.append(openai_client.embeddings.create(
                model=embedding_model_oai, input=np.array(inputs)[index].tolist(), encoding_format="float"
        ))
        response[-1] = [r.embedding for r in response[-1].data]
    return flatten(response)