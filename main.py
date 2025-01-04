from transformers import AutoTokenizer, AutoModelForCasualLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json



def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_dict({
        "input_text": [entry["question"] for entry in data],
        "output_text": [entry["answer"] for entry in data]
    })



dataset = load_custom_dataset('dataset.jsonl')


model_name = 'DeepPavlov/rubert-base-cased'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)