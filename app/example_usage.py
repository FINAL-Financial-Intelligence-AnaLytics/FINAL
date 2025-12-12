from datetime import datetime, timedelta
from llm_client import (
    FinancialConsultationModule,
    FinancialDiagnosisModule,
    GoalsInvestmentsModule
)
from llm_client.implementations import MistralFinancialClient


class SimpleRAGRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        return [
            {
                "content": "Финансовая подушка безопасности - это резервный фонд на случай непредвиденных расходов.",
                "source": "financial_literacy_db"
            }
        ]


def main():
    try:
        llm_client = MistralFinancialClient(
            model_name="mistral-large-latest",
            api_key=None
        )
    except Exception as e:
        print(f"Ошибка инициализации клиента: {e}")
        print("Убедитесь, что установлен mistralai: pip install mistralai")
        print("И установлена переменная окружения MISTRAL_API_KEY")
        return
    
    rag_retriever = SimpleRAGRetriever()
    
    print("=" * 60)
    print("МОДУЛЬ 1: ФИНАНСОВАЯ КОНСУЛЬТАЦИЯ")
    print("=" * 60)
    
    consultation_module = FinancialConsultationModule(
        llm_client=llm_client,
        rag_retriever=rag_retriever
    )
    
    print("\n1. Поиск ответа через RAG:")
    answer = consultation_module.search_answer_via_rag(
        "Что такое финансовая подушка безопасности?"
    )
    print(answer[:200] + "...")
    
    print("\n2. Объяснение термина простым языком:")
    explanation = consultation_module.explain_simple_language("инфляция")
    print(explanation[:200] + "...")
    
    print("\n3. Ответ по финансовой грамотности:")
    literacy_answer = consultation_module.answer_financial_literacy(
        "Как правильно планировать бюджет?"
    )
    print(literacy_answer[:200] + "...")
    
    print("\n4. Обобщенное сравнение продуктов:")
    comparison = consultation_module.compare_financial_products(
        products=["депозит", "облигации", "акции"],
        comparison_criteria=["принцип работы", "уровень риска", "потенциальная доходность"]
    )
    print(comparison[:200] + "...")
    
    print("\n5. Пример расчета:")
    calculation = consultation_module.provide_calculation_example(
        calculation_type="сложный процент",
        parameters={"начальная сумма": 100000, "процент": 10, "лет": 5}
    )
    print(calculation[:200] + "...")
    
    print("\n6. Чек-лист:")
    checklist = consultation_module.provide_checklist("создание финансовой подушки безопасности")
    print(checklist[:200] + "...")
    
    print("\n" + "=" * 60)
    print("МОДУЛЬ 2: ФИНАНСОВАЯ ДИАГНОСТИКА")
    print("=" * 60)
    
    diagnosis_module = FinancialDiagnosisModule(llm_client=llm_client)
    
    print("\n1. Опросник для диагностики:")
    questionnaire = diagnosis_module.get_questionnaire()
    for q in questionnaire[:2]:
        print(f"\nВопрос {q['id']}: {q['question']}")
        if 'options' in q:
            for opt in q['options']:
                print(f"  - {opt}")
    
    print("\n2. Обработка ответов на опросник:")
    responses = {
        1: "Создание финансовой подушки безопасности",
        2: 3,
        3: "10-20%",
        4: "Примерно знаю основные категории расходов",
        5: "Минимальный опыт (депозиты, накопительные счета)"
    }
    profile = diagnosis_module.process_questionnaire_responses(responses)
    print(f"Финансовый стиль: {profile['financial_style']}")
    print(f"Рекомендации: {profile['recommendations']}")
    
    print("\n3. Профиль расходов и доходов:")
    expenses = {
        "Жилье": 30000,
        "Еда": 15000,
        "Транспорт": 5000,
        "Развлечения": 10000,
        "Прочее": 5000
    }
    income = 100000
    expense_profile = diagnosis_module.create_expense_income_profile(expenses, income)
    print(f"Доход: {expense_profile['income']:.2f} руб.")
    print(f"Расходы: {expense_profile['total_expenses']:.2f} руб.")
    print(f"Процент сбережений: {expense_profile['savings_rate']:.2f}%")
    
    print("\n" + "=" * 60)
    print("МОДУЛЬ 3: ЦЕЛИ И ИНВЕСТИЦИИ")
    print("=" * 60)
    
    goals_module = GoalsInvestmentsModule(llm_client=llm_client)
    
    print("\n1. Создание финансовой цели:")
    deadline = datetime.now() + timedelta(days=365)
    goal = goals_module.create_financial_goal(
        title="Накопление на отпуск",
        target_amount=200000,
        deadline=deadline,
        goal_type="накопление",
        current_amount=50000,
        priority=1
    )
    print(f"Цель создана: {goal.title}")
    print(f"Прогресс: {goal.get_progress():.1f}%")
    print(f"Осталось накопить: {goal.get_remaining_amount():.2f} руб.")
    
    print("\n2. План достижения цели:")
    plan = goals_module.generate_achievement_plan(goal, monthly_income=100000)
    print(f"Необходимо откладывать в месяц: {plan['monthly_savings_needed']:.2f} руб.")
    print(f"Процент от дохода: {plan['savings_rate_needed']:.2f}%")
    print(f"Реалистичность: {plan['is_realistic']['realisticity']}")
    
    print("\n3. Все финансовые цели:")
    all_goals = goals_module.get_all_goals()
    for g in all_goals:
        print(f"  - {g.title}: {g.get_progress():.1f}% выполнено")
    
    print("\n" + "=" * 60)
    print("Примеры использования завершены!")
    print("=" * 60)


if __name__ == "__main__":
    main()

