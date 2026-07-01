"""HoneyMoneyDecisionProvider — боевой decision-провайдер команды.

Тонкий оркестратор поверх специализированных модулей (после рефакторинга
god-object на 936 строк разбит на части):

  • market_context.MarketContextFetcher — свечи / новости / IMOEX (сеть + кэш);
  • signals.SignalComputer            — индикаторы + ML prob_up;
  • order_mapper.OrderMapper          — guard-ы и сайзинг ордеров;
  • agents.MultiAgentOrchestrator     — LLM-граф (news/bull/bear/PM);
  • risk_extra.CrisisGate             — стоп открытий при обвале IMOEX.

Поток данных:
  1. DecisionContext (state, positions, trades_today) из main.py
  2. свежие часовые свечи 20 тикеров
  3. indicators + LightGBM×Macro → prob_up
  4. context → MultiAgentOrchestrator → сырые ордера
  5. OrderMapper → типизированные DecisionOrder с риск-сайзингом
  6. CrisisGate отфильтровывает открытия при обвале рынка
  7. DecisionPlan

Безопасность автономного режима:
  • каждый LLM-агент и ML имеют rule-based fallback;
  • любая необработанная ошибка → пустой план + holds для текущих позиций.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.decision.interface import DecisionContext
from src.decision.market_context import MarketContextFetcher
from src.decision.order_mapper import OrderMapper
from src.decision.signals import SignalComputer
from src.decision.strategy_params import StrategyParams
from src.schemas import DecisionOrder, DecisionPlan

logger = logging.getLogger(__name__)

# Список 20 тикеров из ТЗ.
TICKERS_FROM_TZ = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL", "T", "NVTK", "X5",
    "GMKN", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK", "MOEX", "SNGSP", "MTSS", "PIKK",
]

# Дефолты макро-контекста, когда живые данные недоступны (передаются LLM как фон).
_DEFAULT_CBR_RATE = 14.5
_DEFAULT_BRENT_USD = 110.0
_DEFAULT_USD_RUB = 91.0
_DEFAULT_REALIZED_VOL = 0.02


class HoneyMoneyDecisionProvider:
    """Боевой decision-provider команды honey_money_bot."""

    def __init__(
        self,
        data_dir: Path,
        model_path: str = "models/lightgbm_v1.txt",
        macro_priors_path: str = "config/macro_priors.yaml",
        params: Optional[StrategyParams] = None,
        enable_llm: bool = True,
        enable_news: bool = True,
        enable_crisis_gate: bool = True,
        tickers: Optional[List[str]] = None,
        market_data: Optional[Any] = None,
        # Обратная совместимость со старой сигнатурой (main.py мог передавать
        # отдельные числа). Если params не задан — собираем его из этих значений.
        min_cash_reserve: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        order_size_pct: Optional[float] = None,
        default_stop_pct: Optional[float] = None,
        default_take_pct: Optional[float] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.params = params or self._build_params(
            min_cash_reserve, max_position_pct, order_size_pct,
            default_stop_pct, default_take_pct,
        )
        self.enable_llm = enable_llm
        self.enable_news = enable_news
        self.tickers = tickers or TICKERS_FROM_TZ
        self._market_data = market_data

        # Компоненты с graceful fallback.
        self._init_ml(model_path, macro_priors_path)
        self._init_orchestrator()
        self._init_volume_guard()
        candles, news = self._init_data_sources()

        self.market = MarketContextFetcher(
            tickers=self.tickers,
            candles_provider=candles,
            news_aggregator=news,
            market_data=market_data,
            news_fetch_interval_sec=self.params.news_fetch_interval_sec,
            imoex_fetch_interval_sec=self.params.imoex_fetch_interval_sec,
            news_fetch_timeout_sec=self.params.news_fetch_timeout_sec,
        )
        self.signals = SignalComputer(ml_provider=self._ml_provider)
        self.mapper = OrderMapper(params=self.params, tickers=self.tickers)
        self._crisis_gate = self._init_crisis_gate() if enable_crisis_gate else None

        logger.info(
            "HoneyMoneyDecisionProvider initialized: LLM=%s, news=%s, ML=%s, crisis_gate=%s",
            self.enable_llm, self.enable_news,
            "OK" if self._ml_provider is not None else "FALLBACK",
            self._crisis_gate is not None,
        )

    @staticmethod
    def _build_params(min_cash_reserve, max_position_pct, order_size_pct,
                      default_stop_pct, default_take_pct) -> StrategyParams:
        base = StrategyParams()
        return StrategyParams(
            min_cash_reserve=min_cash_reserve if min_cash_reserve is not None else base.min_cash_reserve,
            max_position_pct=max_position_pct if max_position_pct is not None else base.max_position_pct,
            order_size_pct=order_size_pct if order_size_pct is not None else base.order_size_pct,
            default_stop_pct=default_stop_pct if default_stop_pct is not None else base.default_stop_pct,
            default_take_pct=default_take_pct if default_take_pct is not None else base.default_take_pct,
        )

    # ─────────────── Инициализация компонентов ───────────────

    def _init_ml(self, model_path: str, priors_path: str) -> None:
        """LightGBM + macro overlay (с graceful fallback на baseline)."""
        try:
            from src.data.ml_signal import LightGBMMLProvider
            lgb = LightGBMMLProvider(model_path=model_path)
        except Exception as e:
            logger.warning("ML init failed: %s — используем None", e)
            lgb = None
        try:
            from src.data.macro_context import MacroContext
            from src.data.ml_signal import BaselineMLProvider, MacroContextMLProvider
            self.macro = MacroContext(priors_path)
            base = lgb if lgb is not None else BaselineMLProvider()
            self._ml_provider = MacroContextMLProvider(base=base, macro=self.macro)
        except Exception as e:
            logger.warning("Macro init failed: %s", e)
            self.macro = None
            self._ml_provider = None

    def _init_orchestrator(self) -> None:
        """LLM-граф: 4 активных агента (news/bull/bear/PM), остальные на rule-based."""
        self._orchestrator = None
        if not self.enable_llm:
            return
        try:
            from src.agents.orchestrator import MultiAgentOrchestrator
            active = {"news_analyst", "bull_researcher", "bear_researcher", "portfolio_manager"}
            self._orchestrator = MultiAgentOrchestrator(enable_agents=active)
            logger.info("LLM active agents: %s", ", ".join(sorted(active)))
        except Exception as e:
            logger.warning("Orchestrator init failed: %s", e)

    def _init_volume_guard(self) -> None:
        """Отслеживание оборота для требования ≥10М ₽."""
        self._volume_guard = None
        try:
            from src.data.volume_guard import VolumeGuard
            self._volume_guard = VolumeGuard(state_path=str(self.data_dir / "turnover.json"))
        except Exception as e:
            logger.warning("VolumeGuard init failed: %s", e)

    def _init_data_sources(self):
        """Создаёт candles-провайдер и news-агрегатор (оба опциональны)."""
        candles = None
        try:
            from src.data.moex_candles import MoexCandlesProvider
            candles = MoexCandlesProvider(lookback_hours=120)
        except Exception as e:
            logger.warning("Candles provider init failed: %s", e)

        news = None
        if self.enable_news:
            try:
                from src.data.news.aggregator import NewsAggregator
                news = NewsAggregator()
                logger.info("NewsAggregator initialized with keyless RSS sources")
            except Exception as e:
                logger.warning("NewsAggregator init failed: %s — headlines будут пустыми", e)
        return candles, news

    def _init_crisis_gate(self):
        """CrisisGate — блокирует ОТКРЫТИЕ позиций при обвале IMOEX (риск-снижение)."""
        try:
            from src.risk_extra.crisis_gate import CrisisGate
            return CrisisGate(state_path=str(self.data_dir / "crisis_state.json"))
        except Exception as e:
            logger.warning("CrisisGate init failed: %s", e)
            return None

    # ─────────────── Основной decide() ───────────────

    def decide(self, context: DecisionContext) -> DecisionPlan:
        """Главный entry-point. Никогда не должен бросать исключений."""
        t0 = time.time()
        try:
            return self._decide_impl(context)
        except Exception as e:
            logger.exception("HoneyMoney decide failed: %s", e)
            return DecisionPlan(
                orders=[],
                no_action_tickers=list(context.state.positions.keys()),
                cycle_summary=f"Error in decision provider: {e}. Safe mode active.",
                mode="error_fallback",
                raw={"error": str(e)},
            )
        finally:
            logger.info("HoneyMoney decide() elapsed: %.2fs", time.time() - t0)

    def _decide_impl(self, context: DecisionContext) -> DecisionPlan:
        candles_by_ticker = self.market.fetch_candles()
        indicators = self.signals.compute_indicators(candles_by_ticker)
        ml_signals = self.signals.compute_ml_signals(candles_by_ticker)
        portfolio = self._build_portfolio_pct(context)

        imoex = self.market.fetch_imoex_returns()
        orch_context = self._build_orchestrator_context(
            indicators, ml_signals, portfolio, imoex,
        )
        llm_orders, llm_summary, fallback_count = self._run_orchestrator(orch_context)

        decision_orders = self.mapper.map_orders(
            llm_orders, context, candles_by_ticker, self.signals.expected_vols,
        )
        decision_orders, crisis_active = self._apply_crisis_gate(decision_orders, imoex)

        return DecisionPlan(
            orders=decision_orders,
            no_action_tickers=[tk for tk in self.tickers
                               if tk not in {o.ticker for o in decision_orders}],
            cycle_summary=llm_summary or "honey_money cycle",
            next_cycle_in_minutes=15,
            mode="honey_money_multi_agent",
            raw={
                "llm_fallback_count": fallback_count,
                "candles_fetched": sum(1 for v in candles_by_ticker.values() if v),
                "tickers_analyzed": len(indicators),
                "portfolio_pct": portfolio,
                "crisis_active": crisis_active,
            },
        )

    # ─────────────── Вспомогательные шаги ───────────────

    def _apply_crisis_gate(self, orders: List[DecisionOrder], imoex: Dict[str, float]):
        """При обвале IMOEX отбрасываем ОТКРЫВАЮЩИЕ ордера, закрытия пропускаем."""
        if self._crisis_gate is None:
            return orders, False
        try:
            last_close = imoex.get("last_close")
            ret_1d = imoex.get("1d", 0.0)
            # Реконструируем 2-точечный ряд из дневного return для CrisisGate.
            if last_close:
                prev = last_close / (1.0 + ret_1d) if (1.0 + ret_1d) else last_close
                self._crisis_gate.update([prev, last_close])
            if not self._crisis_gate.is_buy_blocked():
                return orders, False
            from src.schemas import DecisionAction
            opening = {DecisionAction.OPEN_LONG, DecisionAction.OPEN_SHORT, DecisionAction.INCREASE}
            filtered = [o for o in orders if o.action not in opening]
            if len(filtered) != len(orders):
                logger.warning("CrisisGate active: dropped %d opening orders",
                               len(orders) - len(filtered))
            return filtered, True
        except Exception as e:
            logger.warning("CrisisGate apply failed: %s", e)
            return orders, False

    def _build_portfolio_pct(self, context: DecisionContext) -> Dict[str, float]:
        """Из текущих позиций и cash считаем долю портфеля по каждому тикеру."""
        total_equity = float(context.state.cash_balance)
        for pos in context.positions:
            avg = context.state.average_prices.get(pos.secid) or pos.average_price or 0.0
            total_equity += abs(pos.position) * avg
        if total_equity <= 0:
            return {}
        return {
            pos.secid: abs(pos.position) * (
                context.state.average_prices.get(pos.secid) or pos.average_price or 0.0
            ) / total_equity
            for pos in context.positions
        }

    @staticmethod
    def _first_positive(*values: Optional[float]) -> float:
        """Первое не-None положительное значение из переданных (live → config → default)."""
        for v in values:
            if v is not None and v > 0:
                return float(v)
        return 0.0

    def _macro_value(self, key: str, default: float) -> float:
        """Читает числовое значение из macro_priors.yaml с безопасным фолбэком."""
        if self.macro and self.macro.loaded:
            try:
                return float(self.macro.state.raw.get(key, default))
            except (TypeError, ValueError):
                return default
        return default

    def _build_orchestrator_context(
        self,
        indicators: Dict[str, Dict[str, Any]],
        ml_signals: Dict[str, float],
        portfolio: Dict[str, float],
        imoex: Dict[str, float],
    ) -> Dict[str, Any]:
        active_flags = []
        if self.macro and self.macro.loaded:
            active_flags = [flag for flag, value in self.macro.state.flags.items() if value]

        headlines = self.market.fetch_news()
        # Приоритет источников: живые котировки MOEX → значение из macro_priors.yaml
        # → жёсткий дефолт-константа. Если MOEX недоступен (вернул None) — мягко
        # откатываемся на конфиг, поэтому бот никогда не остаётся без числа.
        cbr_rate = self._macro_value("cbr_rate_value", _DEFAULT_CBR_RATE)  # ЦБ: только конфиг
        brent_usd = self._first_positive(
            self.market.fetch_brent_usd(),
            self._macro_value("brent_usd_value", _DEFAULT_BRENT_USD),
        )
        usd_rub = self._first_positive(
            self.market.fetch_usd_rub(),
            self._macro_value("usd_rub_value", _DEFAULT_USD_RUB),
        )
        realized_vol = self._first_positive(
            imoex.get("realized_vol"),
            self._macro_value("imoex_realized_vol_value", _DEFAULT_REALIZED_VOL),
        )

        return {
            "imoex_return_5d": imoex.get("5d", 0.0),
            "imoex_return_1d": imoex.get("1d", 0.0),
            "imoex_realized_vol": realized_vol,
            "cbr_rate": cbr_rate,
            "brent_usd": brent_usd,
            "usd_rub": usd_rub,
            "tickers": self.tickers,
            "indicators": indicators,
            "ml_signals": ml_signals,
            "headlines": headlines,
            "top_news_titles": [h["title"] for h in headlines[:8]],
            "active_macro_flags": active_flags,
            "portfolio": portfolio,
        }

    def _run_orchestrator(self, ctx: Dict[str, Any]):
        """Запускает LLM-граф последовательно (sequential безопаснее для rate-limit).

        Возвращает (orders, summary, fallback_count). 9 = «все агенты на fallback».
        """
        if self._orchestrator is None:
            return [], "LLM disabled, no orders", 9
        try:
            run = self._orchestrator.run(ctx)
            return run.orders, run.summary, run.fallback_count
        except Exception as e:
            logger.exception("Orchestrator run failed: %s", e)
            return [], f"Orchestrator error: {e}", 9
