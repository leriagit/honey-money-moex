"""
Agents 6, 7: Bull Researcher vs Bear Researcher (debate).

Снимают характерную проблему LLM в трейдинге — overconfidence.
Bull агрессивно ищет long-идеи, Bear — short-идеи и риски.
Portfolio Manager затем взвешивает.

Используют DeepSeek-V3 — сильное reasoning важнее цены.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import LLMAgent


class BullResearcher(LLMAgent):
    name = "bull_researcher"

    def get_model(self) -> str:
        # Mistral Large 3 (Apache 2.0). Явный id с датой 2512,
        # чтобы избежать алиаса mistral-large = Large 2 (MRL).
        return "mistralai/mistral-large-2512"

    def __init__(self, **kwargs):
        super().__init__(temperature=0.4, max_tokens=1200, **kwargs)

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        regime = context.get("regime_output", {})
        technical = context.get("technical_output", {})
        news = context.get("news_output", {})
        fundamental = context.get("fundamental_output", {})

        system = (
            "Ты Bull Researcher. Твоя задача — найти 3 лучшие LONG-идеи на сегодня "
            "и аргументировать их. Будь конкретен, ссылайся на данные. "
            "Будь умеренно агрессивен, но не overconfident. "
            "Возвращай СТРОГО JSON."
        )
        user = f"""Регим рынка: {regime.get('regime', '?')}, bias: {regime.get('bias', '?')}
Технический analyst summary: {technical.get('summary', '?')}
News overall: {news.get('market_overall', '?')}, narrative: {news.get('narrative', '?')}

Per-ticker данные (выборка):
{self._format_per_ticker(context)}

Найди ТОП-3 LONG-идеи. Верни JSON:
{{
  "ideas": [
    {{ "ticker": "SBER", "conviction": 0..1, "expected_move_pct": 0..10,
       "horizon_hours": 1-24, "reasons": ["причина 1", "причина 2"] }},
    ...
  ],
  "summary": "общая bull thesis 1-2 предложения"
}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _format_per_ticker(ctx: Dict[str, Any]) -> str:
        tech = ctx.get("technical_output", {}).get("per_ticker", {})
        news = ctx.get("news_output", {}).get("per_ticker", {})
        fund = ctx.get("fundamental_output", {}).get("per_ticker", {})
        lines = []
        all_tickers = set(tech.keys()) | set(news.keys()) | set(fund.keys())
        for tk in sorted(all_tickers)[:20]:
            t = tech.get(tk, {}).get("bias", "?")
            n = news.get(tk, {}).get("sentiment", "?")
            f = fund.get(tk, {}).get("bias", "?")
            lines.append(f"  {tk}: tech={t}, news_sent={n}, fundamental={f}")
        return "\n".join(lines) if lines else "(no per-ticker data)"

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Из technical bias извлекаем long-кандидатов
        tech = context.get("technical_output", {}).get("per_ticker", {})
        longs = [
            {"ticker": tk, "conviction": ind.get("strength", 0.5),
             "expected_move_pct": 1.5, "horizon_hours": 4,
             "reasons": [ind.get("reason", "technical long signal")]}
            for tk, ind in tech.items() if ind.get("bias") == "long"
        ][:3]
        return {
            "ideas": longs,
            "summary": "rule-based fallback: long-тикеры из technical analyst",
            "rationale": "fallback",
        }


class BearResearcher(LLMAgent):
    name = "bear_researcher"

    def get_model(self) -> str:
        # Mistral Large 3 (Apache 2.0). Явный id с датой 2512,
        # чтобы избежать алиаса mistral-large = Large 2 (MRL).
        return "mistralai/mistral-large-2512"

    def __init__(self, **kwargs):
        super().__init__(temperature=0.4, max_tokens=1200, **kwargs)

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        regime = context.get("regime_output", {})
        technical = context.get("technical_output", {})
        news = context.get("news_output", {})
        fundamental = context.get("fundamental_output", {})

        system = (
            "Ты Bear Researcher. Твоя задача — найти 3 главных риска или SHORT-идеи "
            "и аргументировать их. Указывай конкретные триггеры падения. "
            "Возвращай СТРОГО JSON."
        )
        user = f"""Регим рынка: {regime.get('regime', '?')}, vol: {regime.get('volatility', '?')}
News narrative: {news.get('narrative', '?')}
Regulator event: {news.get('regulator_event', False)}
RU defense flag: {news.get('ru_defense_flag', False)}

Per-ticker данные (выборка):
{BullResearcher._format_per_ticker(context)}

Найди ТОП-3 SHORT-идеи или риски. Верни JSON:
{{
  "ideas": [
    {{ "ticker": "SBER", "conviction": 0..1, "expected_drop_pct": 0..10,
       "horizon_hours": 1-24, "risks": ["риск 1", "риск 2"] }},
    ...
  ],
  "macro_risks": ["1-2 главных макро-риска на сегодня"],
  "summary": "общая bear thesis 1-2 предложения"
}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tech = context.get("technical_output", {}).get("per_ticker", {})
        shorts = [
            {"ticker": tk, "conviction": ind.get("strength", 0.5),
             "expected_drop_pct": 1.5, "horizon_hours": 4,
             "risks": [ind.get("reason", "technical short signal")]}
            for tk, ind in tech.items() if ind.get("bias") == "short"
        ][:3]
        return {
            "ideas": shorts,
            "macro_risks": ["geopolitical tensions", "regulator events"],
            "summary": "rule-based fallback: short-тикеры из technical analyst",
            "rationale": "fallback",
        }
