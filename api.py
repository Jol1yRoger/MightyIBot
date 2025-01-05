from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify

model_path = "./trained_model"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Введите сообщение"}), 400
    
    inputs = tokenizer.encode(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)