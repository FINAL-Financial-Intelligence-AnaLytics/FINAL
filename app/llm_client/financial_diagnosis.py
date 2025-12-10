from typing import Dict, List, Optional, Any
from enum import Enum
from .base_client import FinancialLLMClient


class FinancialStyle(Enum):
    CONSERVATIVE = "консервативный"
    MODERATE = "умеренный"
    AGGRESSIVE = "агрессивный"
    RISK_AVERSE = "избегающий рисков"
    RISK_TAKING = "принимающий риски"


class FinancialDiagnosisModule:
    def __init__(self, llm_client: FinancialLLMClient):
        self.llm_client = llm_client
        self.questionnaire = self._build_questionnaire()
    
    def _build_questionnaire(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": 1,
                "question": "Какова ваша основная финансовая цель на ближайшие 5 лет?",
                "type": "multiple_choice",
                "options": [
                    "Накопление на крупную покупку",
                    "Создание финансовой подушки безопасности",
                    "Инвестирование для роста капитала",
                    "Планирование выхода на пенсию",
                    "Другое"
                ]
            },
            {
                "id": 2,
                "question": "Как вы относитесь к финансовым рискам?",
                "type": "scale",
                "scale_min": 1,
                "scale_max": 5,
                "scale_labels": {
                    1: "Полностью избегаю рисков",
                    5: "Готов к высоким рискам ради высокой доходности"
                }
            },
            {
                "id": 3,
                "question": "Какой процент от дохода вы обычно откладываете?",
                "type": "multiple_choice",
                "options": [
                    "Менее 10%",
                    "10-20%",
                    "20-30%",
                    "30-50%",
                    "Более 50%"
                ]
            },
            {
                "id": 4,
                "question": "Как вы планируете свой бюджет?",
                "type": "multiple_choice",
                "options": [
                    "Ведю детальный учет всех расходов",
                    "Примерно знаю основные категории расходов",
                    "Не планирую, трачу по ситуации",
                    "Использую финансовые приложения"
                ]
            },
            {
                "id": 5,
                "question": "Какой у вас опыт инвестирования?",
                "type": "multiple_choice",
                "options": [
                    "Нет опыта",
                    "Минимальный опыт (депозиты, накопительные счета)",
                    "Средний опыт (фонды, акции)",
                    "Опытный инвестор"
                ]
            }
        ]
    
    def get_questionnaire(self) -> List[Dict[str, Any]]:
        return self.questionnaire
    
    def process_questionnaire_responses(self, responses: Dict[int, Any]) -> Dict[str, Any]:
        responses_str = self._format_responses(responses)
        
        prompt = f"""Проанализируй ответы пользователя на финансовый опросник:

{responsponses_str}

Определи:
1. Финансовый стиль пользователя (консервативный, умеренный, агрессивный)
2. Основные характеристики финансового поведения
3. Области для улучшения финансовой грамотности
4. Общие рекомендации по развитию финансовых навыков (БЕЗ конкретных продуктов)

ВАЖНО: Не давай персональных инвестиционных советов. Фокусируйся на образовательных рекомендациях.
"""
        
        analysis = self.llm_client.generate_response(prompt)
        financial_style = self._determine_financial_style(responses)
        
        return {
            "financial_style": financial_style.value,
            "analysis": analysis,
            "responses": responses,
            "recommendations": self._generate_educational_recommendations(financial_style)
        }
    
    def create_expense_income_profile(self, expenses: Dict[str, float], income: float) -> Dict[str, Any]:
        total_expenses = sum(expenses.values())
        savings_rate = ((income - total_expenses) / income * 100) if income > 0 else 0
        
        expenses_str = "\n".join(f"- {category}: {amount:.2f} руб." for category, amount in expenses.items())
        
        prompt = f"""Проанализируй финансовый профиль пользователя:

Доход: {income:.2f} руб.
Расходы:
{expenses_str}
Общие расходы: {total_expenses:.2f} руб.
Процент сбережений: {savings_rate:.2f}%

Дай анализ:
1. Структура расходов (какие категории занимают больше всего)
2. Соотношение расходов и доходов
3. Образовательные рекомендации по оптимизации бюджета
4. Принципы управления личными финансами

ВАЖНО: Не давай конкретных советов "куда перевести деньги" или "какой продукт выбрать".
Фокусируйся на принципах и методах управления финансами.
"""
        
        analysis = self.llm_client.generate_response(prompt)
        expense_distribution = {
            category: (amount / total_expenses * 100) if total_expenses > 0 else 0
            for category, amount in expenses.items()
        }
        
        return {
            "income": income,
            "total_expenses": total_expenses,
            "savings_rate": savings_rate,
            "expense_distribution": expense_distribution,
            "analysis": analysis,
            "expenses_by_category": expenses
        }
    
    def determine_financial_style(self, profile_data: Dict[str, Any]) -> FinancialStyle:
        return self._determine_financial_style(profile_data)
    
    def _determine_financial_style(self, data: Dict[str, Any]) -> FinancialStyle:
        risk_tolerance = data.get(2, 3)
        savings_rate = data.get("savings_rate", 0)
        
        if risk_tolerance <= 2:
            return FinancialStyle.CONSERVATIVE
        elif risk_tolerance >= 4:
            return FinancialStyle.AGGRESSIVE
        elif savings_rate >= 30:
            return FinancialStyle.MODERATE
        else:
            return FinancialStyle.CONSERVATIVE
    
    def _format_responses(self, responses: Dict[int, Any]) -> str:
        formatted = []
        for q_id, answer in responses.items():
            question = next((q for q in self.questionnaire if q["id"] == q_id), None)
            if question:
                formatted.append(f"Вопрос {q_id}: {question['question']}\nОтвет: {answer}")
        return "\n\n".join(formatted)
    
    def _generate_educational_recommendations(self, style: FinancialStyle) -> List[str]:
        recommendations_map = {
            FinancialStyle.CONSERVATIVE: [
                "Изучите принципы создания финансовой подушки безопасности",
                "Познакомьтесь с консервативными инструментами сбережения",
                "Изучите основы диверсификации портфеля"
            ],
            FinancialStyle.MODERATE: [
                "Изучите балансирование риска и доходности",
                "Познакомьтесь с различными классами активов",
                "Изучите принципы долгосрочного инвестирования"
            ],
            FinancialStyle.AGGRESSIVE: [
                "Изучите принципы управления рисками",
                "Познакомьтесь с различными стратегиями инвестирования",
                "Изучите основы технического и фундаментального анализа"
            ]
        }
        
        return recommendations_map.get(style, [
            "Изучите основы финансовой грамотности",
            "Познакомьтесь с различными финансовыми инструментами",
            "Развивайте навыки финансового планирования"
        ])

