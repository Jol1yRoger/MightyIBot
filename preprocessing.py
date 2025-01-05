import json


with open('result.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

dataset = []

for chat in data['chats']['list']:
    if 'messages' in chat and chat['type'] != "saved_messages" and chat['name'] != '.':
        messages = chat['messages']

        for i in range(len(messages) - 1):
            current_message = messages[i]
            next_message = messages[i + 1]
            
            if (current_message['type'] == 'message' and next_message['type'] == 'message'
                and 'text' in current_message and 'text' in next_message):
                
                question = current_message['text']
                answer = next_message['text']
                
                if question and answer:
                    dataset.append({"question": question, "answer": answer})

with open('dataset.jsonl', 'w', encoding='utf-8') as outfile:
    for entry in dataset:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Датасет создан. Количество пар: {len(dataset)}")



