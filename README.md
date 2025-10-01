

# Telegram Chatbot with Fine-Tuned ruGPT-3

Проект реализует чат-бота на основе модели **ruGPT-3 Small** от `sberbank-ai`.
Бот обучается на собственных диалогах из Telegram и может работать через **Telegram** и **REST API (Flask)**.

---

## Структура проекта

* **`preprocessing.py`** — извлекает пары *вопрос–ответ* из Telegram-экспорта (`result.json`) и формирует датасет `dataset.jsonl`.
* **`model.py`** — функции для загрузки датасета, препроцессинга и кастомный `Trainer` для обучения модели.
* **`main.py`** — основной скрипт: обучение модели (если её ещё нет) и запуск Telegram-бота.
* **`tg_bot.py`** — логика Telegram-бота: генерация ответов, обработка сообщений.
* **`api.py`** — Flask-сервер с эндпоинтом `/chat` для общения через HTTP-запросы.

---

## Установка

1. Клонируйте проект:

   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

   Минимальный список:

   * `transformers`
   * `datasets`
   * `torch`
   * `scikit-learn`
   * `flask`
   * `python-telegram-bot`

---

## Подготовка данных

1. Экспортируйте переписки из **Telegram Desktop** в JSON (`result.json`).
2. Поместите файл в корень проекта.
3. Запустите:

   ```bash
   python preprocessing.py
   ```

   ➝ Будет создан файл `dataset.jsonl`.

---

## Обучение модели

Запустите:

```bash
python main.py
```

При первом запуске скрипт:

* создаст train/test выборки;
* загрузит модель `ruGPT-3 Small`;
* обучит её на ваших данных;
* сохранит результат в папку `trained_model`.

---

## Telegram-бот

1. Получите токен у `@BotFather`.
2. При запуске скрипта введите токен:

   ```bash
   python main.py
   ```
3. Бот будет отвечать на сообщения.

---

## REST API

Для запуска API:

```bash
python api.py
```

Пример запроса:

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Привет!"}'
```

Ответ:

```json
{"response": "Привет! Рад тебя видеть."}
```


