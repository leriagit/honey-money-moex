"""Сквозной смок-тест провайдера: decide() не падает в degraded-режиме."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.decision.honey_money_provider import HoneyMoneyDecisionProvider
from src.decision.interface import DecisionContext
from src.decision.market_context import MarketContextFetcher
from src.schemas import BotState, Position


def _provider() -> HoneyMoneyDecisionProvider:
    return HoneyMoneyDecisionProvider(
        data_dir=Path(tempfile.mkdtemp()),
        model_path="does-not-exist.txt",
        macro_priors_path="config/macro_priors.yaml",
        enable_llm=False,
        enable_news=False,
        enable_crisis_gate=True,
    )


def test_decide_never_raises_in_degraded_mode():
    prov = _provider()
    st = BotState(cash_balance=500_000, positions={"SBER": 10},
                  average_prices={"SBER": 250.0})
    ctx = DecisionContext(
        state=st,
        positions=[Position(secid="SBER", position=10, average_price=250.0)],
        trades_today=[],
    )
    plan = prov.decide(ctx)
    assert plan.mode in {"honey_money_multi_agent", "error_fallback"}
    assert "crisis_active" in plan.raw


def test_market_context_methods_defined_once():
    """Регрессия: раньше _fetch_news/_imoex были определены дважды (мёртвый код)."""
    fetcher = MarketContextFetcher(tickers=["SBER"], candles_provider=None,
                                   news_aggregator=None)
    # Нет источников → пустые безопасные ответы, без исключений.
    assert fetcher.fetch_candles() == {}
    assert fetcher.fetch_news() == []
    assert isinstance(fetcher.fetch_imoex_returns(), dict)


def test_news_cache_returns_empty_without_aggregator():
    fetcher = MarketContextFetcher(tickers=["SBER"], candles_provider=None,
                                   news_aggregator=None)
    assert fetcher.fetch_news() == []
