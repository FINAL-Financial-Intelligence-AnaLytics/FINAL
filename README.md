# Финансовая RAG Система

Комплексная система финансового ассистента на основе RAG (Retrieval-Augmented Generation) для предоставления образовательных финансовых консультаций.

## Описание

Финансовая RAG система предоставляет интеллектуального ассистента для:

- **Финансовой консультации** - поиск ответов через RAG, объяснения простым языком, обучение финансовой грамотности
- **Финансовой диагностики** - опросники, профили расходов/доходов, определение финансового стиля
- **Планирования целей** - создание финансовых целей, планы достижения, отслеживание прогресса

✅ **Что система МОЖЕТ делать:**
- Обучать финансовой грамотности
- Объяснять экономические понятия простыми словами
- Показывать примеры расчётов
- Давать структуры, чек-листы, алгоритмы
- Объяснять принципы работы инструментов

❌ **Что система НЕ делает:**
- Не даёт персональных инвест-советов
- Не рекомендует продукты
- Не сравнивает "что лучше купить"
- Не даёт юридических или налоговых рекомендаций

### Технологии

- **FastAPI** - веб-фреймворк для API
- **Qdrant** - векторная база данных
- **Mistral AI / OpenRouter** - LLM провайдеры
- **Sentence Transformers** - модели эмбеддингов
- **BM25** - лексический поиск
- **Hybrid Search** - комбинация векторного и лексического поиска

## Установка зависимостей

### Требования

- Python 3.11+
- Docker и Docker Compose (для деплоя)
- Mistral API ключ или OpenRouter API ключ

### Установка Python зависимостей

```bash
# Установка основных зависимостей
pip install -r requirements.txt

# Установка зависимостей для оценки (опционально)
pip install -r requirements-eval.txt
```

### Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```bash
# LLM настройки (выберите один из вариантов)
# Вариант 1: Mistral AI
MISTRAL_API_KEY=your-mistral-api-key
MISTRAL_MODEL=mistral-small-latest
MISTRAL_BASE_URL=https://api.mistral.ai/v1
MISTRAL_TEMPERATURE=0.1

# Вариант 2: OpenRouter
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=tngtech/deepseek-r1t2-chimera:free

# Qdrant настройки
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key  # опционально

# Модель эмбеддингов
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DEVICE=cpu  # или cuda для GPU

# RAG настройки
RAG_TOP_K=5
MAX_TOKENS=2000
TEMPERATURE=0.7
```

## Запуск

### 1. Запуск Qdrant (векторная БД)

```bash
# Запуск Qdrant через Docker
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Или используйте docker-compose из директории `deploy/`:

```bash
cd deploy
docker-compose up -d qdrant
```

### 2. Подготовка данных для RAG

Перед запуском приложения необходимо загрузить данные в Qdrant:

```bash
# Подготовка данных из CSV файла
python scripts/prepare_data.py \
  --input-csv article.pdf  # или путь к вашему CSV файлу
  --collection finance_theory \
  --model intfloat/multilingual-e5-base \
  --qdrant-url http://localhost:6333
```

Параметры скрипта:
- `--input-csv` - путь к CSV файлу с колонкой `text`
- `--collection` - имя коллекции в Qdrant (по умолчанию: `finance_theory`)
- `--model` - модель для генерации эмбеддингов
- `--qdrant-url` - URL Qdrant сервера
- `--recreate` - пересоздать коллекцию (удалит существующие данные)

### 3. Запуск API сервера

#### Локальный запуск

```bash
# Запуск через uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Или через Python
python -m app.main
```

#### Запуск через Gunicorn (продакшн)

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### 4. Проверка работы

После запуска API будет доступен по адресу:
- API: http://localhost:8000
- Документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Пример запроса:

```bash
curl -X POST "http://localhost:8000/api/v1/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Что такое диверсификация портфеля?",
    "limit": 5,
    "score_threshold": 0.2
  }'
```

### 5. Примеры использования

```bash
# Запуск примеров RAG
python -m app.example_rag
```

## Деплой

### Docker Compose (рекомендуется)

Полный стек приложения с Qdrant:

```bash
cd deploy
docker-compose up -d
```

Это запустит:
- **Qdrant** на порту 6333
- **API приложение** на порту 8000

## Оценка качества

Для оценки качества RAG системы:

```bash
python -m app.evaluation
```

Результаты сохраняются в `eval_results/`.