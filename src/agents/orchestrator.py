"""
MultiAgentOrchestrator — координатор графа 9 агентов.

Граф:
                 RegimeAnalyst
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   TechnicalAn     NewsAnalyst    FundamentalAn
        │              │              │
        └──────┬───────┴──────┬───────┘
               ▼              ▼
            PairAnalyst    [merge]
               │              │
               └──────┬───────┘
                      ▼
            BullResearcher ⇄ BearResearcher  (debate)
                      │
                      ▼
                 RiskOfficer
                      │
                      ▼
                PortfolioManager → orders to Executor

Sequential execution (без параллелизма для простоты дебага).
Каждый агент при отказе возвращает fallback — граф НИКОГДА не падает целиком.

Включение/отключение агентов:
  orch = MultiAgentOrchestrator(enable_agents={"regime_analyst", "news_analyst",
                                                "risk_officer", "portfolio_manager"})

Это даёт возможность откатить часть агентов если LLM-бюджет на исходе.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .base import AgentOutput
from .polza_client import PolzaClient
from .regime import RegimeAnalyst
from .news_analyst import NewsAnalyst
from .analysts import TechnicalAnalyst, FundamentalAnalyst, PairAnalyst
from .debate import BullResearcher, BearResearcher
from .risk_pm import RiskOfficer, PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class GraphRun:
    """Результат одного полного прогона графа."""
    ts: str
    orders: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    agent_outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    fallback_count: int = 0
    elapsed_sec: float = 0.0


class MultiAgentOrchestrator:
    """
    Координирует выполнение 9 LLM-агентов и собирает финальное решение.

    Конструктор:
        orch = MultiAgentOrchestrator(
            polza_client=PolzaClient(api_key=os.environ["POLZA_API_KEY"]),
            enable_agents=None,  # None = все 9, иначе — set имён включённых
        )

    Использование:
        result = orch.run(context)
        for order in result.orders:
            executor.send(order)
    """

    DEFAULT_AGENTS = {
        "regime_analyst",
        "technical_analyst",
        "news_analyst",
        "fundamental_analyst",
        "pair_analyst",
        "bull_researcher",
        "bear_researcher",
        "risk_officer",
        "portfolio_manager",
    }

    def __init__(
        self,
        polza_client: Optional[PolzaClient] = None,
        enable_agents: Optional[Set[str]] = None,
    ) -> None:
        self.client = polza_client or PolzaClient()
        enabled = enable_agents or self.DEFAULT_AGENTS

        self.regime = RegimeAnalyst(client=self.client, enable="regime_analyst" in enabled)
        self.technical = TechnicalAnalyst(client=self.client, enable="technical_analyst" in enabled)
        self.news = NewsAnalyst(client=self.client, enable="news_analyst" in enabled)
        self.fundamental = FundamentalAnalyst(client=self.client, enable="fundamental_analyst" in enabled)
        self.pair = PairAnalyst(client=self.client, enable="pair_analyst" in enabled)
        self.bull = BullResearcher(client=self.client, enable="bull_researcher" in enabled)
        self.bear = BearResearcher(client=self.client, enable="bear_researcher" in enabled)
        self.risk = RiskOfficer(client=self.client, enable="risk_officer" in enabled)
        self.pm = PortfolioManager(client=self.client, enable="portfolio_manager" in enabled)

    def run(self, context: Dict[str, Any]) -> GraphRun:
        """
        Главная точка входа. Выполняет граф последовательно.

        Args:
            context: должен содержать:
              - imoex_return_5d, imoex_return_1d, imoex_realized_vol
              - cbr_rate, brent_usd, usd_rub
              - tickers: List[str]
              - indicators: Dict[ticker, IndicatorBundle.dict()]
              - headlines: List[{title, source, ...}]
              - active_macro_flags: List[str]
              - portfolio: Dict[ticker, pct_of_equity]
        """
        import time
        t0 = time.time()
        run = GraphRun(ts=datetime.now(timezone.utc).isoformat())

        # Пауза между LLM-вызовами — даёт DeepInfra rate-limit-у отдохнуть
        RATE_LIMIT_PAUSE = 2.0

        def _maybe_sleep_after_llm(out):
            """Если агент реально звал LLM (не fallback) — пауза перед следующим."""
            if not out.used_fallback:
                time.sleep(RATE_LIMIT_PAUSE)

        # ─── Stage 1: Regime ───
        regime_out = self.regime.analyze(context)
        run.agent_outputs["regime"] = regime_out
        context["regime_output"] = regime_out.data
        _maybe_sleep_after_llm(regime_out)

        # ─── Stage 2 (parallel в дизайне, sequential для простоты): Technical, News, Fundamental ───
        tech_out = self.technical.analyze(context)
        run.agent_outputs["technical"] = tech_out
        context["technical_output"] = tech_out.data
        _maybe_sleep_after_llm(tech_out)

        news_out = self.news.analyze(context)
        run.agent_outputs["news"] = news_out
        context["news_output"] = news_out.data
        _maybe_sleep_after_llm(news_out)

        fund_out = self.fundamental.analyze(context)
        run.agent_outputs["fundamental"] = fund_out
        context["fundamental_output"] = fund_out.data
        _maybe_sleep_after_llm(fund_out)

        # ─── Stage 3: Pair ───
        pair_out = self.pair.analyze(context)
        _maybe_sleep_after_llm(pair_out)
        run.agent_outputs["pair"] = pair_out
        context["pair_output"] = pair_out.data

        # ─── Stage 4: Bull vs Bear ───
        bull_out = self.bull.analyze(context)
        run.agent_outputs["bull"] = bull_out
        context["bull_output"] = bull_out.data
        _maybe_sleep_after_llm(bull_out)

        bear_out = self.bear.analyze(context)
        run.agent_outputs["bear"] = bear_out
        context["bear_output"] = bear_out.data
        _maybe_sleep_after_llm(bear_out)

        # ─── Stage 5: Risk Officer ───
        risk_out = self.risk.analyze(context)
        run.agent_outputs["risk"] = risk_out
        context["risk_output"] = risk_out.data
        _maybe_sleep_after_llm(risk_out)

        # ─── Stage 6: Portfolio Manager (финал) ───
        pm_out = self.pm.analyze(context)
        run.agent_outputs["pm"] = pm_out

        run.orders = pm_out.data.get("orders", [])
        run.summary = pm_out.data.get("summary", "")

        # Сводка
        run.fallback_count = sum(1 for o in run.agent_outputs.values() if o.used_fallback)
        run.total_tokens_in = sum(o.tokens_in for o in run.agent_outputs.values())
        run.total_tokens_out = sum(o.tokens_out for o in run.agent_outputs.values())
        run.elapsed_sec = time.time() - t0

        logger.info(
            "MultiAgent run: %d orders, fallbacks=%d/%d, tokens=%d/%d, elapsed=%.2fs",
            len(run.orders), run.fallback_count, len(run.agent_outputs),
            run.total_tokens_in, run.total_tokens_out, run.elapsed_sec,
        )

        return run

    async def async_run(self, context: Dict[str, Any]) -> GraphRun:
        """
        Async версия run() — независимые ветки графа выполняются ПАРАЛЛЕЛЬНО.

        Stages:
          1. Regime (sequential)
          2. Technical + News + Fundamental (PARALLEL — независимы)
          3. Pair (sequential)
          4. Bull + Bear (PARALLEL — debate, в первом раунде независимы)
          5. RiskOfficer (sequential — нужны Bull+Bear)
          6. PortfolioManager (sequential — финал)

        Ожидаемое ускорение: 134s → ~80s (на 40%) благодаря параллелизму
        в Stage 2 и Stage 4 (самые длинные).
        """
        import asyncio
        import time
        t0 = time.time()
        run = GraphRun(ts=datetime.now(timezone.utc).isoformat())

        # ─── Stage 1: Regime (нужен ДО всех остальных) ───
        regime_out = await self.regime.async_analyze(context)
        run.agent_outputs["regime"] = regime_out
        context["regime_output"] = regime_out.data

        # ─── Stage 2: Technical + News + Fundamental ПАРАЛЛЕЛЬНО ───
        tech_task = self.technical.async_analyze(context)
        news_task = self.news.async_analyze(context)
        fund_task = self.fundamental.async_analyze(context)
        tech_out, news_out, fund_out = await asyncio.gather(tech_task, news_task, fund_task)

        run.agent_outputs["technical"] = tech_out
        run.agent_outputs["news"] = news_out
        run.agent_outputs["fundamental"] = fund_out
        context["technical_output"] = tech_out.data
        context["news_output"] = news_out.data
        context["fundamental_output"] = fund_out.data

        # ─── Stage 3: Pair (мог бы зависеть от Technical, держим sequential) ───
        pair_out = await self.pair.async_analyze(context)
        run.agent_outputs["pair"] = pair_out
        context["pair_output"] = pair_out.data

        # ─── Stage 4: Bull + Bear ПАРАЛЛЕЛЬНО (debate, в первом раунде независимы) ───
        bull_task = self.bull.async_analyze(context)
        bear_task = self.bear.async_analyze(context)
        bull_out, bear_out = await asyncio.gather(bull_task, bear_task)

        run.agent_outputs["bull"] = bull_out
        run.agent_outputs["bear"] = bear_out
        context["bull_output"] = bull_out.data
        context["bear_output"] = bear_out.data

        # ─── Stage 5: Risk Officer ───
        risk_out = await self.risk.async_analyze(context)
        run.agent_outputs["risk"] = risk_out
        context["risk_output"] = risk_out.data

        # ─── Stage 6: Portfolio Manager (финал) ───
        pm_out = await self.pm.async_analyze(context)
        run.agent_outputs["pm"] = pm_out

        run.orders = pm_out.data.get("orders", [])
        run.summary = pm_out.data.get("summary", "")

        run.fallback_count = sum(1 for o in run.agent_outputs.values() if o.used_fallback)
        run.total_tokens_in = sum(o.tokens_in for o in run.agent_outputs.values())
        run.total_tokens_out = sum(o.tokens_out for o in run.agent_outputs.values())
        run.elapsed_sec = time.time() - t0

        logger.info(
            "MultiAgent async_run: %d orders, fallbacks=%d/%d, tokens=%d/%d, elapsed=%.2fs",
            len(run.orders), run.fallback_count, len(run.agent_outputs),
            run.total_tokens_in, run.total_tokens_out, run.elapsed_sec,
        )

        return run

    def run_parallel(self, context: Dict[str, Any]) -> GraphRun:
        """
        Sync обёртка над async_run() — для удобного вызова из обычного кода
        без необходимости управлять asyncio loop вручную.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Уже в async-контексте → fail loudly, нужно использовать async_run напрямую
                raise RuntimeError("run_parallel вызван из async-контекста, используй async_run")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.async_run(context))
