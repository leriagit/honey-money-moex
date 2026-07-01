"""
Agents 8, 9: Risk Officer + Portfolio Manager.

Risk Officer:
  - детерминированные gates (max position, sector cap, drawdown limit)
  - LLM-explainer оборачивает их в читаемый текст
  - имеет право veto на ордер

Portfolio Manager:
  - финальный agent в графе
  - принимает все output'ы предыдущих агентов
  - выдаёт JSON ордеров для Executor: ticker, side, qty_pct, stop, take, rationale
  - использует DeepSeek-V3 (важнее точность чем цена)
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import LLMAgent


class RiskOfficer(LLMAgent):
    """
    Детерминированная риск-проверка + LLM объяснение.

    Чтобы быть надёжным: даже без LLM проверка ДОЛЖНА работать через fallback.
    Поэтому здесь fallback не "пустой" — он содержит реальную логику.
    """
    name = "risk_officer"

    def get_model(self) -> str:
        return "deepseek-chat"

    def __init__(self, max_position_pct: float = 0.20, max_sector_pct: float = 0.50, **kwargs):
        # Снижено с 0.30/0.60 после инцидента MOEX-лонг (25% портфеля).
        # Теперь Risk Officer veto'ит идеи >20% на тикер и >50% на сектор.
        super().__init__(**kwargs)
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        bull_ideas = context.get("bull_output", {}).get("ideas", [])
        bear_ideas = context.get("bear_output", {}).get("ideas", [])
        portfolio = context.get("portfolio", {})  # {ticker: pct_of_equity}
        regime = context.get("regime_output", {})

        system = (
            "Ты Risk Officer. Проверяй идеи bull/bear через гейты риска "
            "и объясняй своё решение. Возвращай JSON."
        )
        user = f"""Текущий портфель (% от equity): {portfolio}
Регим: {regime.get('regime', '?')}, vol: {regime.get('volatility', '?')}
Bull ideas: {bull_ideas}
Bear ideas: {bear_ideas}

Лимиты:
- max position на тикер: {self.max_position_pct*100:.0f}%
- max sector exposure: {self.max_sector_pct*100:.0f}%

Верни JSON:
{{
  "approved_longs": [{{"ticker": "SBER", "max_size_pct": 0..20, "stop_loss_pct": -3, "take_profit_pct": +5}}],
  "approved_shorts": [],
  "vetoed": [{{"ticker": "SBER", "side": "long", "reason": "..."}}],
  "concerns": ["1-2 главных риска"]
}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """РЕАЛЬНАЯ детерминированная риск-логика."""
        bull = context.get("bull_output", {}).get("ideas", [])
        bear = context.get("bear_output", {}).get("ideas", [])
        portfolio = context.get("portfolio", {})

        approved_longs = []
        for idea in bull:
            tk = idea.get("ticker")
            if tk and portfolio.get(tk, 0) < self.max_position_pct:
                approved_longs.append({
                    "ticker": tk,
                    # 10% максимум за один проход — иначе одна идея может выжрать весь лимит
                    "max_size_pct": min(0.10, self.max_position_pct - portfolio.get(tk, 0)) * 100,
                    "stop_loss_pct": -2.4,   # -20% от прежних -3
                    "take_profit_pct": 1.5,  # снижено до +1.5% (быстрее фиксация)
                })

        # ВНИМАНИЕ (намеренная асимметрия, НЕ баг):
        # rule-based fallback Risk Officer не одобряет шорты — это консервативный
        # путь на случай полного отказа LLM. Сами шорты в системе ВКЛЮЧЕНЫ и идут
        # двумя другими путями: (1) PortfolioManager.fallback открывает шорты по
        # ML prob_up<0.43; (2) OrderMapper.map_action маппит SELL_* в OPEN_SHORT.
        # Здесь шорты выключены сознательно, чтобы при тотальном фоллбэке (нет ни
        # LLM, ни ML-сигналов) бот не шортил вслепую. Менять — только вместе со
        # стратегией.
        approved_shorts: list = []
        vetoed = [{"ticker": i.get("ticker"), "side": "short", "reason": "shorts disabled in fallback"}
                  for i in bear]

        return {
            "approved_longs": approved_longs,
            "approved_shorts": approved_shorts,
            "vetoed": vetoed,
            "concerns": ["rule-based fallback Risk Officer"],
            "rationale": "fallback",
        }


class PortfolioManager(LLMAgent):
    """Финальный агент: выдаёт ордера для Executor."""
    name = "portfolio_manager"

    def get_model(self) -> str:
        # Mistral Large 3 2512 (Apache 2.0) — основная модель.
        # Явный id с датой, чтобы избежать алиаса mistral-large = Large 2 (MRL).
        # DeepSeek-V3 (MIT) в ALLOWED_MODELS как резервная.
        return "mistralai/mistral-large-2512"

    def __init__(self, **kwargs):
        super().__init__(temperature=0.1, max_tokens=2048, **kwargs)

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        risk = context.get("risk_output", {})
        portfolio = context.get("portfolio", {})
        regime = context.get("regime_output", {})
        ml_signals = context.get("ml_signals", {}) or {}
        imoex_5d = context.get("imoex_return_5d", 0.0)
        imoex_1d = context.get("imoex_return_1d", 0.0)

        # News context — критически важно для bearish-маркета
        news_output = context.get("news_output", {}) or {}
        market_overall_sentiment = news_output.get("market_overall", 0.0)
        regulator_event = news_output.get("regulator_event", False)
        ru_defense = news_output.get("ru_defense_flag", False)
        news_narrative = news_output.get("narrative", "")
        per_ticker_news = news_output.get("per_ticker", {}) or {}

        # Bull/Bear debate
        bull_output = context.get("bull_output", {}) or {}
        bear_output = context.get("bear_output", {}) or {}
        bull_ideas = bull_output.get("ideas", [])
        bear_ideas = bear_output.get("ideas", [])

        # Топ-7 тикеров с самым ярким sentiment (по модулю)
        news_sorted = sorted(
            per_ticker_news.items(),
            key=lambda kv: abs(float(kv[1].get("sentiment", 0)) * kv[1].get("confidence", 0)),
            reverse=True,
        )[:7]
        news_text = "\n".join(
            f"  {tk}: sentiment={d.get('sentiment', 0):+.2f} "
            f"conf={d.get('confidence', 0):.2f} — {d.get('reason', '')[:80]}"
            for tk, d in news_sorted
        ) or "  (no per-ticker news sentiment)"

        # Quantitative ML signal по тикерам — LightGBM prob_up
        # Показываем топ-7 strongest (по модулю отклонения от 0.5)
        ml_sorted = sorted(
            ml_signals.items(), key=lambda kv: abs(float(kv[1]) - 0.5), reverse=True
        )[:7]
        ml_text = "\n".join(
            f"  {tk}: prob_up={p:.2f} {'(bullish)' if p > 0.55 else '(bearish)' if p < 0.45 else '(neutral)'}"
            for tk, p in ml_sorted
        ) or "  (ML signals unavailable)"

        system = (
            "Ты Portfolio Manager — финальное звено multi-agent графа. "
            "Принимаешь решения учитывая ОДНОВРЕМЕННО: "
            "(1) NEWS sentiment по каждому тикеру от NewsAnalyst LLM "
            "(2) ML LightGBM prob_up по тикерам "
            "(3) Bull/Bear debate — top long/short идеи "
            "(4) Risk Officer одобренные лимиты "
            "(5) Текущий портфель и режим рынка. "
            "ПРАВИЛО ПРИОРИТЕТА: NEWS sentiment >= ML prob_up. "
            "Если news_sentiment < -0.3 — НЕ открывай long даже при высоком prob_up. "
            "Если news_sentiment > +0.3 — НЕ открывай short даже при низком prob_up. "
            "На bearish рынке (IMOEX вниз + market_overall_sentiment<-0.2) — активно SHORTS. "
            "Стратегия mean-reversion: stop -2.4%, take +1.5%. "
            "Отвечай НА РУССКОМ. Возвращай СТРОГО JSON."
        )
        user = f"""Текущий портфель (% от equity): {portfolio}
Рыночный режим: {regime.get('regime')}, vol: {regime.get('volatility')}, bias: {regime.get('bias')}
IMOEX за 5 дней: {imoex_5d*100:+.2f}%, за 1 день: {imoex_1d*100:+.2f}%

📰 NEWS overall: market_sentiment={market_overall_sentiment:+.2f}, regulator_event={regulator_event}, ru_defense={ru_defense}
Narrative: {news_narrative[:200]}

📰 Per-ticker news sentiment (топ-7 сильных):
{news_text}

🤖 ML LightGBM prob_up (топ-7 уверенных):
{ml_text}

🐂 Bull Researcher идеи (top long): {[{'ticker': i.get('ticker'), 'conviction': i.get('conviction')} for i in bull_ideas[:5]]}
🐻 Bear Researcher идеи (top short): {[{'ticker': i.get('ticker'), 'conviction': i.get('conviction')} for i in bear_ideas[:5]]}

✅ Risk Officer одобрил:
- longs: {risk.get('approved_longs', [])[:5]}
- shorts: {risk.get('approved_shorts', [])[:5]}

Сформируй ордера используя ВСЕ источники. Логика:
  • news_sentiment < -0.3 AND prob_up < 0.5 → STRONG SELL (открыть short)
  • news_sentiment < -0.2 → закрыть существующие long, не открывать новые
  • news_sentiment > +0.3 AND prob_up > 0.55 → STRONG BUY (open_long)
  • Bull-идея + news_sentiment > 0 + prob_up > 0.55 → BUY
  • Bear-идея + news_sentiment < 0 → SELL (короткий шорт)
  • Конфликт сигналов → HOLD

Верни JSON:
{{
  "orders": [
    {{
      "ticker": "SBER",
      "action": "BUY"|"BUY_SMALL"|"HOLD"|"SELL_HALF"|"SELL_ALL",
      "size_pct": 0..20,
      "stop_loss_pct": -2.4,
      "take_profit_pct": 1.5,
      "rationale": "ссылайся на news_sentiment, prob_up И Bull/Bear-аргументы"
    }}
  ],
  "summary": "1-2 предложения общего плана"
}}

ПОМНИ: SELL_HALF/SELL_ALL без позиции открывает SHORT.
НЕ открывай long на bearish news (как ГМК отказался от дивидендов, эскалация СВО)."""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Маппим Risk Officer approved + ML prob_up → orders.

        Без LLM: используем ml_signals для фильтра:
          - prob_up > 0.55 → BUY_SMALL
          - prob_up < 0.45 → SELL_HALF (→ OPEN_SHORT в адаптере)
          - 0.45..0.55 → HOLD
        """
        risk = context.get("risk_output", {})
        ml_signals = context.get("ml_signals", {}) or {}
        orders = []

        # Approved longs от Risk Officer + ML prob_up фильтр
        for long_idea in risk.get("approved_longs", []):
            tk = long_idea["ticker"]
            prob_up = float(ml_signals.get(tk, 0.5))
            if prob_up < 0.5:
                continue  # ML против — пропускаем
            orders.append({
                "ticker": tk,
                "action": "BUY_SMALL",
                "size_pct": long_idea.get("max_size_pct", 15),
                "stop_loss_pct": long_idea.get("stop_loss_pct", -2.4),
                "take_profit_pct": 1.5,  # -20% от прежних настроек, быстрее фиксация
                "rationale": f"fallback: одобрено Risk Officer, ML prob_up={prob_up:.2f}",
            })

        # SHORT-идеи на основе ML: для тикеров с prob_up < 0.45 (если bull-кандидатов мало)
        # Это включает shorts даже когда Risk Officer не дал short-ideas через rule-based fallback
        if len(orders) < 3:
            short_candidates = [
                (tk, p) for tk, p in ml_signals.items()
                if float(p) < 0.43
            ]
            short_candidates.sort(key=lambda x: x[1])  # самые слабые впереди
            for tk, p in short_candidates[:3 - len(orders)]:
                orders.append({
                    "ticker": tk,
                    "action": "SELL_HALF",  # маппится в OPEN_SHORT адаптером
                    "size_pct": 10,
                    "stop_loss_pct": -2.4,
                    "take_profit_pct": 1.5,
                    "rationale": f"fallback short: ML prob_up={p:.2f} (low)",
                })

        return {
            "orders": orders,
            "summary": "rule-based fallback Portfolio Manager",
            "rationale": "fallback",
        }
