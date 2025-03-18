# CARROT: A Cost Aware Rate Optimal Router

Welcome to our GitHub repository! This repository is based on the ideas introduced in

[Somerstep, Seamus, Felipe Maia Polo, Allysson Flavio Melo de Oliveira, Prattyush Mangal, Mírian Silva, Onkar Bhardwaj, Mikhail Yurochkin, and Subha Maity. "CARROT: A Cost Aware Rate Optimal Router." arXiv preprint arXiv:2502.03261 (2025).](https://arxiv.org/abs/2502.03261)

The repository is still under construction and only part of experiments/plots of the paper can be currently reproduced. See TODO section below.

## Overview

CARROT is a lightweight, efficient, and theoretically optimal router designed for intelligently directing queries to Large Language Models (LLMs). With the rapid expansion of available LLMs, selecting the most cost-effective model capable of producing an adequate response is increasingly critical. CARROT selects models by balancing performance and cost, leveraging robust statistical estimates to ensure optimal routing decisions. It is computationally efficient and proven minimax rate-optimal, providing confidence in both cost savings and performance quality.

### Key Features
- **Cost-Aware Selection:** Routes queries to the cheapest capable LLM, optimizing for customizable cost-performance trade-offs.
- **Minimax Optimality:** Theoretically established to achieve rate-optimal routing performance.
- **Computational Efficiency:** Lightweight design ensures rapid decision-making with minimal computational overhead.

## Quick Start

Run our [Google Colab demo](https://github.com/somerstep/CARROT/blob/main/notebooks/CARROT_KNN_demo.ipynb) and learn how to use CARROT-KNN for simple, well-performing, and efficient routing of LMs!

## Installation

To use the code in this repository, clone the repo and create a conda environment using:

```
conda env create --file=environment.yml
conda activate carrot
pip install jupyter notebook
```

##  Smart Price-aware Routing (SPROUT) dataset and pre-trained router

To access our routing dataset, SPROUT, and our pre-trained router, please check [CARROT's HuggingFace page](https://huggingface.co/CARROT-LLM-Routing).


## Reproducing results from the paper

1. Insert your OPENAI API key on `carrot/data_utils.py`.
2. Generate data by running `gen_open-llm-lb-v2.py` and `gen_sprout.py`.
3. Train prediction models by running `train_and_infer.py`. You will have to specify which dataset you want to work with, e.g., `python train_and_infer.py --dataset 'open-llm-lb-v2'`.
4. Activate Jupyter Notebook and run `carrot/plots.ipynb` to generate plots.

## TODOs
1. Include results for RouterBench.
2. Include rest of paper plots (e.g., choice of models).
3. Create demo for router training. 

## Citing

```
@article{somerstep2025carrot,
  title={CARROT: A Cost Aware Rate Optimal Router},
  author={Somerstep, Seamus and Maia Polo, Felipe and de Oliveira, Allysson Flavio Melo and Mangal, Prattyush and Silva, M{\'\i}rian and Bhardwaj, Onkar and Yurochkin, Mikhail and Maity, Subha},
  journal={arXiv preprint arXiv:2502.03261},
  year={2025}
}
```
