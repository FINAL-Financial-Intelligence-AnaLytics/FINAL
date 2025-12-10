from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from .base_client import FinancialLLMClient


class FinancialGoal:
    def __init__(
        self,
        goal_id: str,
        title: str,
        target_amount: float,
        current_amount: float,
        deadline: datetime,
        goal_type: str,
        priority: int = 1
    ):
        self.goal_id = goal_id
        self.title = title
        self.target_amount = target_amount
        self.current_amount = current_amount
        self.deadline = deadline
        self.goal_type = goal_type
        self.priority = priority
    
    def get_progress(self) -> float:
        if self.target_amount == 0:
            return 0.0
        return min((self.current_amount / self.target_amount) * 100, 100.0)
    
    def get_remaining_amount(self) -> float:
        return max(self.target_amount - self.current_amount, 0.0)
    
    def get_months_remaining(self) -> int:
        now = datetime.now()
        if self.deadline < now:
            return 0
        delta = self.deadline - now
        return max(int(delta.days / 30), 1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "target_amount": self.target_amount,
            "current_amount": self.current_amount,
            "deadline": self.deadline.isoformat(),
            "goal_type": self.goal_type,
            "priority": self.priority,
            "progress": self.get_progress(),
            "remaining_amount": self.get_remaining_amount(),
            "months_remaining": self.get_months_remaining()
        }


class GoalsInvestmentsModule:
    def __init__(self, llm_client: FinancialLLMClient):
        self.llm_client = llm_client
        self.goals: Dict[str, FinancialGoal] = {}
    
    def create_financial_goal(
        self,
        title: str,
        target_amount: float,
        deadline: datetime,
        goal_type: str = "накопление",
        current_amount: float = 0.0,
        priority: int = 1
    ) -> FinancialGoal:
        goal_id = f"goal_{len(self.goals) + 1}_{datetime.now().timestamp()}"
        
        goal = FinancialGoal(
            goal_id=goal_id,
            title=title,
            target_amount=target_amount,
            current_amount=current_amount,
            deadline=deadline,
            goal_type=goal_type,
            priority=priority
        )
        
        self.goals[goal_id] = goal
        return goal
    
    def generate_achievement_plan(self, goal: FinancialGoal, monthly_income: float) -> Dict[str, Any]:
        remaining_amount = goal.get_remaining_amount()
        months_remaining = goal.get_months_remaining()
        monthly_savings_needed = remaining_amount / months_remaining if months_remaining > 0 else 0
        savings_rate_needed = (monthly_savings_needed / monthly_income * 100) if monthly_income > 0 else 0
        
        prompt = f"""Создай план достижения финансовой цели:

Название цели: {goal.title}
Целевая сумма: {goal.target_amount:.2f} руб.
Текущая сумма: {goal.current_amount:.2f} руб.
Осталось накопить: {remaining_amount:.2f} руб.
Срок: {months_remaining} месяцев
Месячный доход: {monthly_income:.2f} руб.
Необходимо откладывать в месяц: {monthly_savings_needed:.2f} руб. ({savings_rate_needed:.2f}% от дохода)

Создай структурированный план, который включает:
1. Анализ реалистичности цели (достижима ли она при текущих условиях)
2. Рекомендации по оптимизации бюджета (общие принципы, БЕЗ конкретных продуктов)
3. Стратегию накопления (принципы, методы)
4. Чек-лист действий
5. Образовательные материалы по теме

ВАЖНО:
- Не рекомендую конкретные финансовые продукты
- Фокусируйся на принципах и методах
- Объясняй, как работает накопление и планирование
- Давай образовательную информацию
"""
        
        plan_text = self.llm_client.generate_response(prompt)
        plan = {
            "goal": goal.to_dict(),
            "plan_text": plan_text,
            "monthly_savings_needed": monthly_savings_needed,
            "savings_rate_needed": savings_rate_needed,
            "months_remaining": months_remaining,
            "is_realistic": self._assess_realisticity(monthly_savings_needed, monthly_income),
            "milestones": self._generate_milestones(goal, months_remaining),
            "strategies": self._generate_savings_strategies(goal.goal_type)
        }
        
        return plan
    
    def update_goal_progress(self, goal_id: str, new_amount: float) -> Optional[FinancialGoal]:
        if goal_id not in self.goals:
            return None
        
        goal = self.goals[goal_id]
        goal.current_amount = new_amount
        return goal
    
    def get_all_goals(self) -> List[FinancialGoal]:
        return list(self.goals.values())
    
    def get_goal(self, goal_id: str) -> Optional[FinancialGoal]:
        return self.goals.get(goal_id)
    
    def delete_goal(self, goal_id: str) -> bool:
        if goal_id in self.goals:
            del self.goals[goal_id]
            return True
        return False
    
    def compare_goals(self, goal_ids: List[str]) -> str:
        goals = [self.goals[gid] for gid in goal_ids if gid in self.goals]
        
        if not goals:
            return "Цели не найдены."
        
        goals_info = "\n".join([
            f"- {goal.title}: {goal.target_amount:.2f} руб., срок: {goal.get_months_remaining()} мес., приоритет: {goal.priority}"
            for goal in goals
        ])
        
        prompt = f"""Проанализируй следующие финансовые цели и дай рекомендации по приоритизации:

{goals_info}

Дай анализ:
1. Какие цели более срочные
2. Какие цели более реалистичны
3. Общие принципы приоритизации финансовых целей
4. Стратегия достижения нескольких целей одновременно

ВАЖНО: Не давай конкретных советов "какую цель выбрать". Объясняй принципы приоритизации.
"""
        
        return self.llm_client.generate_response(prompt)
    
    def _assess_realisticity(self, monthly_savings_needed: float, monthly_income: float) -> Dict[str, Any]:
        savings_rate = (monthly_savings_needed / monthly_income * 100) if monthly_income > 0 else 100
        
        if savings_rate <= 20:
            realisticity = "реалистично"
        elif savings_rate <= 40:
            realisticity = "требует оптимизации"
        else:
            realisticity = "требует пересмотра"
        
        return {
            "realisticity": realisticity,
            "savings_rate": savings_rate,
            "recommendation": self._get_realisticity_recommendation(savings_rate)
        }
    
    def _get_realisticity_recommendation(self, savings_rate: float) -> str:
        if savings_rate <= 20:
            return "Цель достижима при разумном планировании бюджета."
        elif savings_rate <= 40:
            return "Цель требует оптимизации расходов и поиска дополнительных источников дохода."
        else:
            return "Рекомендуется пересмотреть срок или сумму цели, либо найти способы увеличения дохода."
    
    def _generate_milestones(self, goal: FinancialGoal, months: int) -> List[Dict[str, Any]]:
        milestones = []
        remaining = goal.get_remaining_amount()
        for i in range(1, 5):
            milestone_amount = goal.current_amount + (remaining * i / 4)
            milestone_month = int(months * i / 4)
            milestones.append({
                "milestone": f"{i * 25}%",
                "target_amount": milestone_amount,
                "target_month": milestone_month,
                "description": f"Достижение {i * 25}% от цели"
            })
        
        return milestones
    
    def _generate_savings_strategies(self, goal_type: str) -> List[str]:
        strategies_map = {
            "накопление": [
                "Принцип 'сначала заплати себе' - откладывайте сразу после получения дохода",
                "Автоматизация накоплений через отдельный счет",
                "Постепенное увеличение процента откладываемых средств"
            ],
            "покупка": [
                "Изучите принципы сравнения цен и планирования крупных покупок",
                "Рассмотрите возможность накопления с учетом инфляции",
                "Изучите методы оптимизации бюджета для ускорения накоплений"
            ],
            "инвестирование": [
                "Изучите принципы диверсификации портфеля",
                "Познакомьтесь с различными классами активов",
                "Изучите основы долгосрочного инвестирования"
            ]
        }
        
        return strategies_map.get(goal_type, [
            "Регулярное откладывание фиксированной суммы",
            "Отслеживание прогресса и корректировка плана",
            "Изучение принципов финансового планирования"
        ])

