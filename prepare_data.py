import json
from datasets import load_dataset

dataset = load_dataset('vicgalle/alpaca-gpt4', split="train")

with open('alpaca-gpt4.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')