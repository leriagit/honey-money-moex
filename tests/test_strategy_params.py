"""Тесты единого конфига стратегии."""
from __future__ import annotations

from src.decision.strategy_params import StrategyParams


def test_defaults_consistent():
    p = StrategyParams()
    # stop жёстче take (mean-reversion профиль)
    assert p.default_stop_pct == 0.024
    assert p.default_take_pct == 0.015
    assert 0 < p.max_position_pct <= 1


def test_hard_max_order_derived_from_capital():
    p = StrategyParams(starting_capital_rub=500_000, max_position_pct=0.20)
    assert p.hard_max_order_rub == 100_000


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("DEFAULT_STOP_PCT", "0.05")
    monkeypatch.setenv("ORDER_SIZE_PCT", "0.25")
    p = StrategyParams.from_env()
    assert p.default_stop_pct == 0.05
    assert p.order_size_pct == 0.25


def test_from_env_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("DEFAULT_STOP_PCT", "not-a-number")
    p = StrategyParams.from_env()
    assert p.default_stop_pct == StrategyParams().default_stop_pct
