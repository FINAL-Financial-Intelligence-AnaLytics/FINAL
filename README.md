# Финансовый RAG LLM Клиент

Система финансового ассистента на основе RAG (Retrieval-Augmented Generation) с модульной архитектурой для различных финансовых задач.

## Описание

Финансовый LLM клиент предоставляет три основных модуля:

1. **Финансовая консультация** - поиск ответов через RAG, объяснения простым языком, обучение финансовой грамотности
2. **Финансовая диагностика** - опросники, профили расходов/доходов, определение финансового стиля
3. **Цели и инвестиции** - создание финансовых целей, планы достижения

## Возможности ассистента

✅ **Что ассистент МОЖЕТ делать:**
- Обучать финансовой грамотности
- Объяснять экономические понятия простыми словами
- Показывать примеры расчётов
- Давать структуры, чек-листы, алгоритмы
- Объяснять принципы работы инструментов

❌ **Что ассистент НЕ делает:**
- Не даёт персональных инвест-советов
- Не рекомендует продукты
- Не сравнивает "что лучше купить"
- Не даёт юридических или налоговых рекомендаций

## Структура проекта

```
rag/
├── app/
│   ├── llm_client/
│   │   ├── __init__.py                 # Экспорт основных классов
│   │   ├── base_client.py              # Базовый класс LLM клиента
│   │   ├── financial_consultation.py   # Модуль финансовой консультации
│   │   ├── financial_diagnosis.py     # Модуль финансовой диагностики
│   │   ├── goals_investments.py       # Модуль целей и инвестиций
│   │   └── implementations/
│   │       ├── __init__.py
│   │       └── openai_client.py       # Реализация для OpenAI
│   └── example_usage.py               # Примеры использования
├── requirements.txt                    # Зависимости проекта
└── README.md                          # Документация
```

## Установка

1. Клонируйте репозиторий или скопируйте файлы проекта

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Установите переменную окружения для API ключа (для OpenAI):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Или создайте файл `.env`:
```
OPENAI_API_KEY=your-api-key-here
```

## Быстрый старт

### Базовое использование

```python
from llm_client import (
    FinancialConsultationModule,
    FinancialDiagnosisModule,
    GoalsInvestmentsModule
)
from llm_client.implementations import OpenAIFinancialClient
from datetime import datetime, timedelta

# Инициализация LLM клиента
llm_client = OpenAIFinancialClient(
    model_name="gpt-3.5-turbo",
    api_key=None  # Будет использован из переменной окружения
)

# ========== МОДУЛЬ 1: ФИНАНСОВАЯ КОНСУЛЬТАЦИЯ ==========
consultation = FinancialConsultationModule(
    llm_client=llm_client,
    rag_retriever=None  # Опционально: ваш RAG retriever
)

# Поиск ответа через RAG
answer = consultation.search_answer_via_rag(
    "Что такое финансовая подушка безопасности?"
)

# Объяснение термина простым языком
explanation = consultation.explain_simple_language("инфляция")

# Ответ по финансовой грамотности
literacy_answer = consultation.answer_financial_literacy(
    "Как правильно планировать бюджет?"
)

# Обобщенное сравнение продуктов
comparison = consultation.compare_financial_products(
    products=["депозит", "облигации", "акции"]
)

# Пример расчета
calculation = consultation.provide_calculation_example(
    calculation_type="сложный процент",
    parameters={"начальная сумма": 100000, "процент": 10, "лет": 5}
)

# Чек-лист
checklist = consultation.provide_checklist(
    "создание финансовой подушки безопасности"
)

# ========== МОДУЛЬ 2: ФИНАНСОВАЯ ДИАГНОСТИКА ==========
diagnosis = FinancialDiagnosisModule(llm_client=llm_client)

# Получение опросника
questionnaire = diagnosis.get_questionnaire()

# Обработка ответов
responses = {
    1: "Создание финансовой подушки безопасности",
    2: 3,  # Уровень риска
    3: "10-20%",
    4: "Примерно знаю основные категории расходов",
    5: "Минимальный опыт"
}
profile = diagnosis.process_questionnaire_responses(responses)

# Профиль расходов/доходов
expenses = {
    "Жилье": 30000,
    "Еда": 15000,
    "Транспорт": 5000,
    "Развлечения": 10000
}
income = 100000
expense_profile = diagnosis.create_expense_income_profile(expenses, income)

# ========== МОДУЛЬ 3: ЦЕЛИ И ИНВЕСТИЦИИ ==========
goals = GoalsInvestmentsModule(llm_client=llm_client)

# Создание цели
deadline = datetime.now() + timedelta(days=365)
goal = goals.create_financial_goal(
    title="Накопление на отпуск",
    target_amount=200000,
    deadline=deadline,
    current_amount=50000
)

# План достижения
plan = goals.generate_achievement_plan(goal, monthly_income=100000)

# Получение всех целей
all_goals = goals.get_all_goals()
```

## Интеграция с RAG системой

Для использования поиска через RAG, необходимо реализовать класс retriever с методом `retrieve()`:

```python
class YourRAGRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        """
        Поиск релевантных документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество документов для возврата
            
        Returns:
            Список словарей с ключами 'content' и опционально 'source'
        """
        # Ваша логика поиска в векторной БД
        return [
            {
                "content": "Текст документа...",
                "source": "название_источника"
            }
        ]

# Использование
rag_retriever = YourRAGRetriever()
consultation = FinancialConsultationModule(
    llm_client=llm_client,
    rag_retriever=rag_retriever
)
```

## Модули

### 1. FinancialConsultationModule

Модуль финансовой консультации предоставляет:

- `search_answer_via_rag(query, top_k)` - поиск ответов через RAG
- `explain_simple_language(term, context)` - объяснение терминов простым языком
- `answer_financial_literacy(question)` - ответы по финансовой грамотности
- `compare_financial_products(products, criteria)` - обобщенное сравнение продуктов
- `provide_calculation_example(calculation_type, parameters)` - примеры расчетов
- `provide_checklist(topic)` - чек-листы по темам

### 2. FinancialDiagnosisModule

Модуль финансовой диагностики предоставляет:

- `get_questionnaire()` - получение опросника
- `process_questionnaire_responses(responses)` - обработка ответов и создание профиля
- `create_expense_income_profile(expenses, income)` - анализ расходов и доходов
- `determine_financial_style(profile_data)` - определение финансового стиля

### 3. GoalsInvestmentsModule

Модуль целей и инвестиций предоставляет:

- `create_financial_goal(...)` - создание финансовой цели
- `generate_achievement_plan(goal, monthly_income)` - план достижения цели
- `update_goal_progress(goal_id, new_amount)` - обновление прогресса
- `get_all_goals()` - получение всех целей
- `compare_goals(goal_ids)` - сравнение и приоритизация целей

## Создание собственной реализации LLM клиента

Для использования другого LLM провайдера, создайте класс, наследующийся от `FinancialLLMClient`:

```python
from llm_client.base_client import FinancialLLMClient

class YourLLMClient(FinancialLLMClient):
    def generate_response(self, prompt: str, context=None):
        # Ваша реализация вызова LLM API
        response = your_llm_api_call(prompt)
        return response
```

## Примеры использования

Полные примеры использования всех модулей находятся в файле `app/example_usage.py`.

Запуск примеров:
```bash
cd app
python example_usage.py
```

## Безопасность и ограничения

Система автоматически проверяет запросы на соответствие правилам работы ассистента и добавляет дисклеймеры к ответам. Все ответы носят образовательный характер и не являются персональными финансовыми рекомендациями.

## Лицензия

[Укажите вашу лицензию]

## Контакты

[Ваши контакты]

