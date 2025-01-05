from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_response(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


tg_token = input("Enter telegram token: ")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я чат-бот. Просто напиши мне сообщение, и я отвечу!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    try:
        response = generate_response(user_input)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")