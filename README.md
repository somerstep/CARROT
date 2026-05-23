# CARROT: A Cost Aware Rate Optimal Router

Code and data-prep pipeline for

[Somerstep, Seamus, Felipe Maia Polo, Allysson Flavio Melo de Oliveira, Prattyush Mangal, Mírian Silva, Onkar Bhardwaj, Mikhail Yurochkin, and Subha Maity. "CARROT: A Cost Aware Rate Optimal Router." arXiv preprint arXiv:2502.03261 (2025).](https://arxiv.org/abs/2502.03261)

## Overview

CARROT is a lightweight, efficient, and theoretically optimal router for directing queries to Large Language Models. With the rapid expansion of available LLMs, selecting the cheapest model capable of producing an adequate response is increasingly critical. CARROT picks models by balancing predicted performance and cost, leveraging robust statistical estimates to make optimal routing decisions. It is computationally efficient and minimax rate-optimal.

### Key Features
- **Cost-aware selection:** routes queries to the cheapest capable LLM, with a tunable cost / performance trade-off (`λ`).
- **Minimax optimality:** rate-optimal routing performance.
- **Lightweight:** KNN and RoBERTa variants both run on a laptop.

## Repository layout

```
carrot/         # training + inference package
  gen_routerbench.py        # build train/test splits for Routerbench
  gen_open-llm-lb-v2.py     # build train/test splits for Open-LLM-Leaderboard v2
  gen_sprout.py             # build train/test splits for SPROUT
  train_and_infer.py        # fit routers, write per-router predictions to data/{ds}/preds/
  utils.py, constants.py, data_utils.py
demo/
  router.py                 # minimal CarrotRouter wrapping the released HF checkpoints
notebooks/
  plot1_sprout_spider.ipynb         # Plot 1: CARROT vs gpt-4o on SPROUT
  plot2_routerbench_binary.ipynb    # Plot 2: CARROT vs binary routers on Routerbench
  plot3_routerbench_vs_rb.ipynb     # Plot 3: CARROT vs Routerbench router on Routerbench
  plot4_sprout_vs_rb_zero.ipynb     # Plot 4: CARROT vs Routerbench + zero router on SPROUT
  plot5_openllm_binary.ipynb        # Plot 5: CARROT vs binary routers on Open-LLM-Leaderboard v2
  plots_bw.ipynb                    # Print-friendly black-and-white versions of plots 1–5
  knn_accuracy.ipynb                # CARROT-KNN test-split metrics across all three datasets
  sprout_embedding_dim.ipynb        # PCA / Kernel PCA / Isomap on SPROUT OpenAI embeddings
plots/          # rendered PDFs + knn_accuracy_table.md
archive/        # earlier NeurIPS/ICML scratch (kept for reference)
```

## Quick start (use the released router)

The fastest path is the wrapper in `demo/router.py`, which loads the published RoBERTa checkpoints (`CARROT-LLM-Routing/Performance` and `CARROT-LLM-Routing/Cost`) from HuggingFace:

```python
from demo.router import CarrotRouter

router = CarrotRouter(hf_token="hf_...")
selected = router.route(["Explain entropy in one sentence."], mu=0.3)
print(selected)
```

`mu` ∈ [0, 1] tunes the cost / performance trade-off (0 = always the most accurate model, 1 = always the cheapest).

## Installation

```
conda env create --file=environment.yml      # Linux / CUDA
# or
conda env create --file=environment-macos.yml   # macOS / MPS
conda activate carrot
pip install jupyter notebook
```

Copy `.env.example` → `.env` and fill in `OPENAI_API_KEY` (used by `gen_routerbench.py` and `gen_open-llm-lb-v2.py` for `text-embedding-3-small`). `HF_TOKEN` is only needed for gated models.

## SPROUT dataset and pre-trained router

The SPROUT dataset and the pre-trained RoBERTa routers are on the [CARROT HuggingFace page](https://huggingface.co/CARROT-LLM-Routing).

## Reproducing the paper plots

1. Build per-dataset train/test splits (one-time, requires the OpenAI key):
   ```
   cd carrot
   python gen_routerbench.py
   python gen_open-llm-lb-v2.py
   python gen_sprout.py
   ```
2. Fit routers and write per-router predictions into `data/{dataset}/preds/`:
   ```
   python train_and_infer.py --dataset routerbench   --routers carrot-knn carrot-roberta mf rorf roberta-binary
   python train_and_infer.py --dataset open-llm-lb-v2 --routers carrot-knn carrot-roberta mf rorf roberta-binary
   python train_and_infer.py --dataset sprout         --routers carrot-knn
   ```
   (`--trainer custom` uses the bundled PyTorch loop, which is the right choice on Apple Silicon; `--trainer hf` uses HuggingFace `Trainer`.)
3. Render the figures:
   ```
   cd ../notebooks
   jupyter nbconvert --to notebook --execute plot1_sprout_spider.ipynb --inplace
   jupyter nbconvert --to notebook --execute plot2_routerbench_binary.ipynb --inplace
   jupyter nbconvert --to notebook --execute plot3_routerbench_vs_rb.ipynb --inplace
   jupyter nbconvert --to notebook --execute plot4_sprout_vs_rb_zero.ipynb --inplace
   jupyter nbconvert --to notebook --execute plot5_openllm_binary.ipynb --inplace
   ```
   PDFs land in `plots/`.

## Citing

```
@article{somerstep2025carrot,
  title={CARROT: A Cost Aware Rate Optimal Router},
  author={Somerstep, Seamus and Maia Polo, Felipe and de Oliveira, Allysson Flavio Melo and Mangal, Prattyush and Silva, M{\'\i}rian and Bhardwaj, Onkar and Yurochkin, Mikhail and Maity, Subha},
  journal={arXiv preprint arXiv:2502.03261},
  year={2025}
}
```
