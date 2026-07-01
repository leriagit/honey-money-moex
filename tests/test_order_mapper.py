"""Тесты OrderMapper — ядро торговых guard-ов и сайзинга."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.decision.interface import DecisionContext
from src.decision.order_mapper import OrderMapper
from src.decision.strategy_params import StrategyParams
from src.schemas import AgentOrder, BotState, DecisionAction, Position, Trade

TICKERS = ["SBER", "GAZP", "LKOH"]


@pytest.fixture
def mapper() -> OrderMapper:
    return OrderMapper(params=StrategyParams(), tickers=TICKERS)


def _ctx(cash=500_000.0, positions=None, avg=None, trades=None) -> DecisionContext:
    positions = positions or []
    state = BotState(
        cash_balance=cash,
        positions={p.secid: p.position for p in positions},
        average_prices=avg or {},
    )
    return DecisionContext(state=state, positions=positions, trades_today=trades or [])


def _candles(price: float, n: int = 5) -> list:
    return [{"begin": "2026-05-25T10:00:00", "open": price, "high": price,
             "low": price, "close": price, "volume": 1000} for _ in range(n)]


# ─────────────── map_action таблица ───────────────

@pytest.mark.parametrize("action,qty,expected", [
    ("BUY", 0, DecisionAction.OPEN_LONG),
    ("BUY", 10, DecisionAction.INCREASE),
    ("BUY", -10, DecisionAction.CLOSE),
    ("SELL_ALL", 0, DecisionAction.OPEN_SHORT),
    ("SELL_ALL", 10, DecisionAction.CLOSE),
    ("SELL_ALL", -10, DecisionAction.INCREASE),
    ("SELL_HALF", 10, DecisionAction.REDUCE),
    ("SELL_HALF", -10, None),
    ("SELL_HALF", 0, DecisionAction.OPEN_SHORT),
    ("HOLD", 0, None),
    ("UNKNOWN", 0, None),
])
def test_map_action_table(action, qty, expected):
    assert OrderMapper.map_action(action, qty) == expected


# ─────────────── anti-wash-trade ───────────────

def test_dedup_wash_trades_keeps_first(mapper):
    orders = [
        AgentOrder(ticker="SBER", action="BUY"),
        AgentOrder(ticker="SBER", action="SELL_ALL"),  # встречный → отбрасывается
    ]
    deduped = mapper._dedup_wash_trades(orders)
    assert len(deduped) == 1
    assert deduped[0].action == "BUY"


def test_dedup_drops_hold(mapper):
    assert mapper._dedup_wash_trades([AgentOrder(ticker="SBER", action="HOLD")]) == []


def test_agent_order_from_raw_tolerant():
    assert AgentOrder.from_raw({"ticker": "SBER", "action": "BUY"}).ticker == "SBER"
    assert AgentOrder.from_raw({"action": "BUY"}) is None      # нет тикера
    assert AgentOrder.from_raw("garbage") is None              # не dict
    # уже типизированный объект проходит насквозь
    ao = AgentOrder(ticker="GAZP")
    assert AgentOrder.from_raw(ao) is ao


# ─────────────── открытие лонга и hard cap ───────────────

def test_open_long_basic(mapper):
    ctx = _ctx()
    orders = mapper.map_orders(
        [{"ticker": "SBER", "action": "BUY", "size_pct": 10}],
        ctx, {"SBER": _candles(250.0)},
    )
    assert len(orders) == 1
    o = orders[0]
    assert o.action == DecisionAction.OPEN_LONG
    assert o.size_lots >= 1
    # hard cap: одна сделка не дороже 20% от 500К = 100К ₽
    assert o.size_lots * 250.0 <= StrategyParams().hard_max_order_rub * 1.1


def test_hard_cap_limits_huge_order(mapper):
    # size_pct=90% от 490К usable → должно упереться в hard cap 100К
    ctx = _ctx()
    orders = mapper.map_orders(
        [{"ticker": "SBER", "action": "BUY", "size_pct": 90}],
        ctx, {"SBER": _candles(100.0)},
    )
    assert orders[0].size_lots * 100.0 <= StrategyParams().hard_max_order_rub * 1.1


# ─────────────── min-hold guard ───────────────

def test_min_hold_blocks_recent_ticker(mapper):
    recent = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=3)))
    trade = Trade(
        tradedate=recent.strftime("%Y-%m-%d"),
        tradetime=recent.strftime("%H:%M:%S"),
        secid="SBER", quantity=10, price=250.0,
    )
    ctx = _ctx(trades=[trade])
    orders = mapper.map_orders(
        [{"ticker": "SBER", "action": "BUY"}],
        ctx, {"SBER": _candles(250.0)},
    )
    assert orders == []  # заблокировано min-hold


# ─────────────── anti-averaging guard ───────────────

def test_anti_averaging_down_blocks_increase(mapper):
    pos = [Position(secid="SBER", position=10, average_price=300.0)]
    ctx = _ctx(positions=pos, avg={"SBER": 300.0})
    # цена 250 < avg 300 → докупать лонг нельзя
    orders = mapper.map_orders(
        [{"ticker": "SBER", "action": "BUY", "size_pct": 10}],
        ctx, {"SBER": _candles(250.0)},
    )
    assert orders == []


def test_increase_allowed_above_avg(mapper):
    pos = [Position(secid="SBER", position=10, average_price=200.0)]
    ctx = _ctx(positions=pos, avg={"SBER": 200.0})
    orders = mapper.map_orders(
        [{"ticker": "SBER", "action": "BUY", "size_pct": 10}],
        ctx, {"SBER": _candles(250.0)},
    )
    assert len(orders) == 1
    assert orders[0].action == DecisionAction.INCREASE


# ─────────────── фильтр чужих тикеров ───────────────

def test_unknown_ticker_skipped(mapper):
    ctx = _ctx()
    orders = mapper.map_orders(
        [{"ticker": "AAPL", "action": "BUY"}],
        ctx, {"AAPL": _candles(100.0)},
    )
    assert orders == []
