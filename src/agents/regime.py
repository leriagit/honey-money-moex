"""Agent 1: Market Regime Analyst.

Определяет режим рынка (bull/range/bear) и желаемый bias.
Использует Qwen 2.5 14B — быстрая и дешёвая модель для классификации.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import LLMAgent


class RegimeAnalyst(LLMAgent):
    name = "regime_analyst"

    def get_model(self) -> str:
        # Изначально планировался qwen-2.5-14b (быстрый/дешёвый),
        # но в нашей Polza-подписке доступен только deepseek-chat.
        return "deepseek-chat"

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        imoex_ret_5d = context.get("imoex_return_5d", 0.0)
        imoex_ret_1d = context.get("imoex_return_1d", 0.0)
        vol = context.get("imoex_realized_vol", 0.0)
        cbr_rate = context.get("cbr_rate", 14.5)
        brent = context.get("brent_usd", 110.0)
        macro_flags = context.get("active_macro_flags", [])
        top_news = context.get("top_news_titles", [])[:5]

        system = (
            "Ты Market Regime Analyst для торгового бота на Московской бирже (MOEX). "
            "Анализируй макро-контекст и определяй режим рынка. "
            "Отвечай СТРОГО валидным JSON, без markdown-обёрток."
        )
        user = f"""Контекст рынка MOEX:
- IMOEX за 5 дней: {imoex_ret_5d:+.2%}
- IMOEX за 1 день: {imoex_ret_1d:+.2%}
- Реализованная волатильность: {vol:.3f}
- Ключевая ставка ЦБ РФ: {cbr_rate}%
- Brent: ${brent:.1f}
- Активные макро-флаги: {', '.join(macro_flags) if macro_flags else 'нет'}
- Топ-5 новостей: {'; '.join(top_news) if top_news else 'нет'}

Верни JSON формата:
{{
  "regime": "bull" | "range" | "bear",
  "volatility": "low" | "mid" | "high",
  "bias": "long_heavy" | "long_light" | "neutral" | "short_light" | "short_heavy",
  "confidence": 0.0-1.0,
  "key_drivers": ["драйвер1", "драйвер2"],
  "narrative": "1-2 предложения почему именно такой режим"
}}"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based: по знаку и величине 5d return + волатильности."""
        ret_5d = context.get("imoex_return_5d", 0.0)
        vol = context.get("imoex_realized_vol", 0.02)

        if ret_5d > 0.03:
            regime, bias = "bull", "long_heavy"
        elif ret_5d > 0.01:
            regime, bias = "bull", "long_light"
        elif ret_5d < -0.03:
            regime, bias = "bear", "short_light"
        elif ret_5d < -0.01:
            regime, bias = "range", "neutral"
        else:
            regime, bias = "range", "neutral"

        vol_label = "high" if vol > 0.03 else "mid" if vol > 0.015 else "low"

        return {
            "regime": regime,
            "volatility": vol_label,
            "bias": bias,
            "confidence": 0.5,
            "key_drivers": [f"IMOEX 5d={ret_5d:+.2%}", f"vol={vol:.3f}"],
            "narrative": "rule-based fallback: классификация по 5-дневному return",
            "rationale": "fallback",
        }
