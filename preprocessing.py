import json

# Загрузка файла
with open('result.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Инициализация списка для датасета
dataset = []

# Проходимся по всем чатам
for chat in data['chats']['list']:
    if 'messages' in chat and chat['type'] != "saved_messages":
        messages = chat['messages']
        # Генерация пар вопрос-ответ
        for i in range(len(messages) - 1):
            current_message = messages[i]
            next_message = messages[i + 1]
            
            # Учитываем только текстовые сообщения
            if (current_message['type'] == 'message' and next_message['type'] == 'message'
                and 'text' in current_message and 'text' in next_message):
                
                # Обработка текстовых сообщений
                question = current_message['text']
                answer = next_message['text']
                
                # Исключение пустых текстов
                if question and answer:
                    dataset.append({"question": question, "answer": answer})

# Сохранение датасета
with open('dataset.jsonl', 'w', encoding='utf-8') as outfile:
    for entry in dataset:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Датасет создан. Количество пар: {len(dataset)}")



