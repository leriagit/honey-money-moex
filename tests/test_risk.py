"""Тесты детерминированного pre-trade risk gate."""
from __future__ import annotations

import pytest

from src.risk import PreTradeValidator, RiskConfig
from src.schemas import BotState, OrderRequest, OrderSide, Position


@pytest.fixture
def validator() -> PreTradeValidator:
    return PreTradeValidator(RiskConfig())


def _req(direction="B", qty=10, secid="SBER"):
    return OrderRequest(direction=OrderSide(direction), secid=secid, quantity=qty, bot="b")


def test_buy_within_limits_allowed(validator):
    r = validator.validate(_req(), 50_000, 500_000, BotState(cash_balance=500_000), [])
    assert r.allowed


def test_order_exceeds_cash_share_rejected(validator):
    r = validator.validate(_req(), 200_000, 500_000, BotState(cash_balance=500_000), [])
    assert not r.allowed
    assert r.reason == "order_value_exceeds_cash_share_limit"


def test_safe_mode_blocks_opening_but_allows_close(validator):
    st = BotState(cash_balance=500_000, safe_mode=True)
    opening = validator.validate(_req(), 50_000, 500_000, st, [], closing_order=False)
    closing = validator.validate(_req(), 50_000, 500_000, st, [], closing_order=True)
    assert not opening.allowed
    assert closing.allowed


def test_cash_reserve_protected(validator):
    # покупка, которая опустит cash ниже резерва 10К
    r = validator.validate(_req(qty=1), 495_000, 500_000, BotState(cash_balance=500_000), [])
    assert not r.allowed


def test_shorts_disabled_blocks_oversell():
    val = PreTradeValidator(RiskConfig(allow_shorts=False))
    positions = [Position(secid="SBER", position=5)]
    r = val.validate(_req(direction="S", qty=10), 50_000, 500_000,
                     BotState(cash_balance=500_000), positions)
    assert not r.allowed
    assert r.reason == "shorts_are_disabled"
