from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json, torch, os

from model import load_custom_dataset, preprocess_data, CustomTrainer

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import AutoTokenizer, AutoModelForCausalLM

from tg_bot import tg_token, CommandHandler, start, handle_message


if __name__ == "__main__":


    if not os.path.isdir('trained_model'):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство: {device}")

        dataset = load_custom_dataset('dataset.jsonl')
        dataset_dict = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

        model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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


    print("Starting telegram bot")

    application = Application.builder().token(tg_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
