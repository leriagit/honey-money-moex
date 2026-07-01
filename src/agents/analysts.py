"""
Агенты 2, 4, 5 — узкоспециализированные аналитики.

  TechnicalAnalyst    — техническая картина (RSI/MACD/SMA → bias по тикерам)
  FundamentalAnalyst  — сектор + макро-контекст (отрасль, дивиденды, мультипликаторы)
  PairAnalyst         — pair trading, cointegration spreads

Все три используют Qwen 2.5 14B (быстрая и дешёвая).
Работают параллельно с RegimeAnalyst.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import LLMAgent


class TechnicalAnalyst(LLMAgent):
    """Аналитик технических индикаторов."""
    name = "technical_analyst"

    def get_model(self) -> str:
        return "deepseek-chat"

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        indicators_per_ticker = context.get("indicators", {})  # {ticker: {rsi, macd_hist, sma_ratio, ...}}
        tickers = list(indicators_per_ticker.keys())
        if not tickers:
            tickers = context.get("tickers", [])

        lines = []
        for tk in tickers[:20]:
            ind = indicators_per_ticker.get(tk, {})
            rsi = ind.get("rsi_14", "?")
            macd_h = ind.get("macd_hist", "?")
            vol_r = ind.get("volume_ratio", "?")
            sma = ind.get("sma20_ratio", "?")
            lines.append(f"{tk}: RSI={rsi}, MACD_hist={macd_h}, vol_ratio={vol_r}, price/SMA20={sma}")

        system = (
            "Ты Technical Analyst для бота на MOEX. По техническим индикаторам "
            "оценивай bias (long/neutral/short) по каждому тикеру. "
            "Возвращай СТРОГО валидный JSON."
        )
        user = f"""Индикаторы по тикерам:
{chr(10).join(lines)}

Верни JSON:
{{
  "per_ticker": {{
    "SBER": {{ "bias": "long"|"neutral"|"short", "strength": 0..1, "reason": "короткое объяснение" }},
    ...
  }},
  "summary": "общая техническая картина 1-2 предложения"
}}"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Простое rule-based: RSI < 30 → long, RSI > 70 → short."""
        per_ticker = {}
        indicators = context.get("indicators", {})
        for tk, ind in indicators.items():
            rsi = ind.get("rsi_14")
            if rsi is None:
                per_ticker[tk] = {"bias": "neutral", "strength": 0.0, "reason": "no data"}
            elif rsi < 30:
                per_ticker[tk] = {"bias": "long", "strength": 0.6, "reason": f"RSI={rsi} oversold"}
            elif rsi > 70:
                per_ticker[tk] = {"bias": "short", "strength": 0.6, "reason": f"RSI={rsi} overbought"}
            else:
                per_ticker[tk] = {"bias": "neutral", "strength": 0.0, "reason": f"RSI={rsi} mid"}
        return {
            "per_ticker": per_ticker,
            "summary": "rule-based RSI threshold",
            "rationale": "fallback",
        }


class FundamentalAnalyst(LLMAgent):
    """Аналитик секторных и фундаментальных факторов."""
    name = "fundamental_analyst"

    def get_model(self) -> str:
        return "deepseek-chat"

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        tickers = context.get("tickers", [])
        cbr_rate = context.get("cbr_rate", 14.5)
        brent = context.get("brent_usd", 110.0)
        usd_rub = context.get("usd_rub", 91.0)
        regime = context.get("regime", "range")

        system = (
            "Ты Fundamental Analyst по MOEX. Оценивай фундаментальную "
            "привлекательность тикеров с учётом секторных факторов. "
            "Возвращай СТРОГО JSON."
        )
        user = f"""Макро-контекст:
- Ключевая ставка ЦБ: {cbr_rate}% (банки: + при снижении; девелопмент: + при снижении)
- Brent: ${brent} (нефтянка: + при росте до $115, - при пробое $120)
- USD/RUB: {usd_rub} (экспортёры металлы/нефть: + при слабом рубле)
- Режим рынка: {regime}

Тикеры: {', '.join(tickers)}

Верни JSON:
{{
  "per_ticker": {{
    "SBER": {{ "bias": "long"|"neutral"|"short", "strength": 0..1, "reason": "..." }},
    ...
  }},
  "sector_summary": {{
    "banks": "...", "oil_gas": "...", "metals": "...", "retail": "...",
    "tech": "...", "telecom": "...", "transport": "...", "developers": "..."
  }}
}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based: используем macro_priors через MacroContext если есть."""
        tickers = context.get("tickers", [])
        return {
            "per_ticker": {tk: {"bias": "neutral", "strength": 0.0, "reason": "fallback"} for tk in tickers},
            "sector_summary": {},
            "rationale": "fallback",
        }


class PairAnalyst(LLMAgent):
    """Pair trading: ищет коинтегрированные пары и оценивает spread."""
    name = "pair_analyst"

    def get_model(self) -> str:
        return "deepseek-chat"

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        # Передаём корреляции пар (если посчитаны заранее)
        pairs = context.get("pair_spreads", [])

        system = (
            "Ты Pair Analyst для MOEX. Ищи возможности pair trading "
            "(коинтегрированные пары с разошедшимся спредом). "
            "Возвращай JSON."
        )
        user = f"""Известные пары и их spreads:
{chr(10).join(f"- {p['pair']}: z-score={p.get('z_score', '?')}, corr={p.get('corr', '?')}" for p in pairs[:10]) if pairs else "(данные о парах не предоставлены)"}

Верни JSON:
{{
  "opportunities": [
    {{ "long": "SBER", "short": "VTBR", "spread_z": 2.1, "reason": "..." }},
    ...
  ]
}}

Если возможностей нет — пустой массив."""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"opportunities": [], "rationale": "fallback"}
