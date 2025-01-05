from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json
import torch



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    


def to_string(value):
        if isinstance(value, list):
            return " ".join(str(item) if isinstance(item, str) else str(item.get("text", "")) for item in value)
        return str(value)


def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [
            {
                "question": to_string(entry["question"]),
                "answer": to_string(entry["answer"])
            }
            for entry in (json.loads(line) for line in f)
        ]
    return Dataset.from_dict({
        "input_text": [entry["question"] for entry in data],
        "output_text": [entry["answer"] for entry in data]
    })


def preprocess_data(examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    inputs = tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = tokenizer(
        examples["output_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )["input_ids"]
    inputs["labels"] = labels

    return inputs