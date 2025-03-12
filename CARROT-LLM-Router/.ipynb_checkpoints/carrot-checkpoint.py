from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class CarrotRouter:
    def __init__(self, hf_token):
        self.hf_token = hf_token

        # Define model costs
        self.COSTS = {
            'aws-claude-3-5-sonnet-v1': [3, 15], 'aws-titan-text-premier-v1': [0.5, 1.5],
            'openai-gpt-4o': [2.5, 10], 'openai-gpt-4o-mini': [0.15, 0.6],
            'wxai-granite-3-2b-instruct-8k-max-tokens': [0.1, 0.1],
            'wxai-granite-3-8b-instruct-8k-max-tokens': [0.2, 0.2],
            'wxai-llama-3-1-70b-instruct': [0.9, 0.9], 'wxai-llama-3-1-8b-instruct': [0.2, 0.2],
            'wxai-llama-3-2-1b-instruct': [0.06, 0.06], 'wxai-llama-3-2-3b-instruct': [0.06, 0.06],
            'wxai-llama-3-3-70b-instruct': [0.9, 0.9], 'wxai-mixtral-8x7b-instruct-v01': [0.6, 0.6],
            'wxai-llama-3-405b-instruct': [3.5, 3.5]
        }

        # Load tokenizers and models
        self.input_counter = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B", token=self.hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        self.score_predictor = AutoModelForSequenceClassification.from_pretrained(
            'CARROT-LLM-Routing/Performance',
            problem_type="multi_label_classification",
            num_labels=len(self.COSTS),
        )

        self.output_counter = AutoModelForSequenceClassification.from_pretrained(
            'CARROT-LLM-Routing/Cost',
            problem_type="regression",
            num_labels=len(self.COSTS),
        )

        # Map index to model names
        self.id2label = list(self.COSTS.keys())

    def route(self, prompts, mu):
        tokenized_text = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            is_split_into_words=False,
            return_tensors='pt'
        )

        self.input_counter.pad_token = self.tokenizer.eos_token

        scores = 1 / (1 + np.exp(-self.score_predictor(tokenized_text["input_ids"]).logits.detach().numpy()))
        output_tokens = self.output_counter(tokenized_text["input_ids"]).logits.detach().numpy()

        input_tokens = [self.input_counter(prompt, return_tensors="pt")["input_ids"].shape[1] for prompt in prompts]
        input_tokens = np.array(input_tokens).T

        costs = []
        for i, model in enumerate(self.COSTS.keys()):
            cost = (input_tokens * self.COSTS[model][0] / 1_000_000) + (output_tokens[:, i] * self.COSTS[model][1] / 1_000)
            costs.append(cost.tolist())

        costs = np.array(costs).T
        model_idx = ((1 - mu) * scores - mu * costs * 100).argmax(axis=1, keepdims=True)
        selected_models = [self.id2label[idx[0]] for idx in model_idx]

        return selected_models



