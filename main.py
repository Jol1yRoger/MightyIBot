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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    def to_string(value):
        if isinstance(value, list):
            return " ".join(str(item) if isinstance(item, str) else str(item.get("text", "")) for item in value)
        return str(value)

    # Загрузка датасета
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

    dataset = load_custom_dataset('dataset.jsonl')

    dataset_dict = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def preprocess_data(examples):
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

    tokenized_train_dataset = train_dataset.map(preprocess_data, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_data, batched=True)

    training_args = TrainingArguments(
        output_dir='./trained_model',
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        fp16=True 
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset
    )


    trainer.train()


    trainer.save_model('./trained_model')
    tokenizer.save_pretrained('./trained_model')
