import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from datasets import load_dataset
from openai import OpenAI
import numpy as np
import tiktoken

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to the project-root .env file "
        "(see .env.example) or export it in your shell."
    )

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
    
def get_oai_embs(inputs, max_batch_size=2000, max_tokens_per_request=250_000):
    """Batch inputs for the OpenAI embeddings API. The API caps both samples (2048)
    and total tokens per request (300k for text-embedding-3-small). We greedily
    pack by token count — measuring with tiktoken — so a batch with a few long
    prompts gets split before the API rejects it."""
    encoding = tiktoken.encoding_for_model(embedding_model_oai)
    token_counts = [len(encoding.encode(x)) for x in inputs]

    batches = []
    cur_idx, cur_tokens = [], 0
    for i, tok in enumerate(token_counts):
        # If a single input exceeds the cap, truncate it upfront.
        if tok > max_tokens_per_request:
            inputs[i] = encoding.decode(encoding.encode(inputs[i])[:max_tokens_per_request])
            tok = max_tokens_per_request
        if cur_idx and (cur_tokens + tok > max_tokens_per_request or len(cur_idx) >= max_batch_size):
            batches.append(cur_idx)
            cur_idx, cur_tokens = [], 0
        cur_idx.append(i)
        cur_tokens += tok
    if cur_idx:
        batches.append(cur_idx)

    response = []
    for idx_batch in batches:
        texts_batch = [inputs[i] for i in idx_batch]
        resp = openai_client.embeddings.create(
            model=embedding_model_oai, input=texts_batch, encoding_format="float"
        )
        response.append([r.embedding for r in resp.data])
    return flatten(response)