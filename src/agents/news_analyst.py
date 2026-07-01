"""Agent 3: News & Sentiment Analyst.

Получает свежие заголовки от NewsAggregator и возвращает per-ticker sentiment.
Заменяет словарный sentiment (RU_KEYWORDS / POSITIVE_WORDS) на реальный LLM.

Использует DeepSeek-V3 — он лучше понимает русский текст и финансовые нюансы.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import LLMAgent


class NewsAnalyst(LLMAgent):
    name = "news_analyst"

    def get_model(self) -> str:
        # Mistral Large 3 (Apache 2.0). Явный id с датой 2512,
        # чтобы избежать алиаса mistral-large = Large 2 (MRL).
        return "mistralai/mistral-large-2512"

    def __init__(self, **kwargs):
        super().__init__(max_tokens=2048, **kwargs)

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        headlines = context.get("headlines", [])
        tickers = context.get("tickers", [])
        if not headlines or not tickers:
            headlines = headlines or [{"title": "no news available"}]

        # 25 заголовков — Mistral Large справится с большим контекстом
        headlines_text = "\n".join(
            f"- [{h.get('source', '?')}] {h.get('title', '')[:150]}"
            for h in headlines[:25]
        )
        tickers_text = ", ".join(tickers)

        system = (
            "Ты News & Sentiment Analyst для торгового бота на Московской бирже (MOEX). "
            "Анализируй финансовые новости НА РУССКОМ ЯЗЫКЕ и определяй sentiment "
            "по каждому тикеру: положительный (+) для bullish-новостей, "
            "отрицательный (-) для bearish, нейтральный (0) если нет связи. "
            "ВАЖНО: ты работаешь в bear-режиме рынка (IMOEX в нисходящем тренде). "
            "Учитывай как явные так и косвенные сигналы:\n"
            "- ГМК Норникель без дивидендов → bearish для GMKN\n"
            "- 'Удары по ВПК Украины', эскалация СВО → bearish для всего рынка, особенно банки/потребсектор\n"
            "- Brent падает → bearish для нефтянки (LKOH/ROSN/SNGSP/NVTK)\n"
            "- ЦБ снижает ставку → bullish для банков (SBER/VTBR/T) и девелоперов (PIKK)\n"
            "- Дивиденды одобрены → bullish для тикера до отсечки\n"
            "- Санкционные риски → bearish для экспортёров (металлы, нефть)\n"
            "Отвечай СТРОГО валидным JSON без markdown."
        )
        user = f"""Тикеры для анализа: {tickers_text}

Свежие заголовки за последние 24ч:
{headlines_text}

Верни JSON формата:
{{
  "per_ticker": {{
    "SBER": {{ "sentiment": -1.0..+1.0, "confidence": 0..1, "reason": "короткое объяснение НА РУССКОМ" }},
    ...
  }},
  "market_overall": -1.0..+1.0,
  "regulator_event": true|false,
  "ru_defense_flag": true|false,
  "narrative": "1-2 предложения общего настроения НА РУССКОМ"
}}

КРИТИЧНО:
- sentiment ∈ [-1, +1]: -1=крайне bearish, 0=нейтрал, +1=крайне bullish
- confidence ∈ [0, 1]: 0=нет связанных новостей, 1=сильный сигнал
- regulator_event=true если ЦБ/Кремль выпустил критическое событие
- ru_defense_flag=true если RU-источники активно защищают негативную новость → подозрение на short
- Заполни ВСЕ {len(tickers)} тикеров. Без новостей — sentiment=0, confidence=0
- Если общий рынок bearish (IMOEX падает) — market_overall ≤ -0.3
- НЕ бойся ставить отрицательный sentiment если новости плохие"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based: используем существующий словарный sentiment из base.py
        Просто возвращаем нейтрал для всех тикеров.
        """
        tickers = context.get("tickers", [])
        return {
            "per_ticker": {
                tk: {"sentiment": 0.0, "confidence": 0.0, "reason": "fallback: no LLM"}
                for tk in tickers
            },
            "market_overall": 0.0,
            "regulator_event": False,
            "ru_defense_flag": False,
            "narrative": "fallback (rule-based): LLM news analyst недоступен",
            "rationale": "fallback",
        }
