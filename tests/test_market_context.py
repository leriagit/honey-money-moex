"""Тесты live-данных MOEX: парсинг ISS-ответов и realized_vol.

Сеть не дёргаем — проверяем чистые функции парсинга на синтетических ответах
ISS той же формы, что отдаёт https://iss.moex.com. Сам сетевой путь имеет
graceful fallback (None → значение из macro_priors.yaml), это покрыто в
test_provider_smoke.
"""
from __future__ import annotations

import pytest

from src.decision.market_context import (
    BRENT_ASSET_CODE,
    MarketContextFetcher,
)


# ─────────────── realized_vol ───────────────

def test_realized_vol_zero_on_flat_series():
    assert MarketContextFetcher.realized_vol([100, 100, 100, 100]) == 0.0


def test_realized_vol_positive_on_moving_series():
    vol = MarketContextFetcher.realized_vol([100, 102, 99, 103, 98])
    assert vol > 0


def test_realized_vol_too_short():
    assert MarketContextFetcher.realized_vol([100]) == 0.0
    assert MarketContextFetcher.realized_vol([]) == 0.0


# ─────────────── _rows ───────────────

def test_rows_zips_columns():
    data = {"candles": {"columns": ["open", "close"], "data": [[1, 2], [3, 4]]}}
    rows = MarketContextFetcher._rows(data, "candles")
    assert rows == [{"open": 1, "close": 2}, {"open": 3, "close": 4}]


def test_rows_missing_block():
    assert MarketContextFetcher._rows(None, "candles") == []
    assert MarketContextFetcher._rows({}, "candles") == []


# ─────────────── _last_candle_close ───────────────

def test_last_candle_close_takes_last_valid():
    data = {"candles": {"columns": ["close"], "data": [[90.0], [91.5], [None]]}}
    # последняя строка close=None → берём предыдущую валидную
    assert MarketContextFetcher._last_candle_close(data) == 91.5


def test_last_candle_close_empty():
    assert MarketContextFetcher._last_candle_close(None) is None


# ─────────────── _pick_front_future ───────────────

def _listing(rows):
    return {"securities": {"columns": ["SECID", "ASSETCODE", "LASTTRADEDATE"], "data": rows}}


def test_pick_front_future_nearest_non_expired():
    data = _listing([
        ["BRM6", BRENT_ASSET_CODE, "2026-06-01"],   # уже истёк
        ["BRQ6", BRENT_ASSET_CODE, "2026-08-31"],   # дальний
        ["BRN6", BRENT_ASSET_CODE, "2026-07-31"],   # ближайший валидный
        ["SiU6", "Si", "2026-07-15"],               # другой ASSETCODE
    ])
    assert MarketContextFetcher._pick_front_future(data, BRENT_ASSET_CODE, "2026-06-25") == "BRN6"


def test_pick_front_future_none_when_all_expired():
    data = _listing([["BRM6", BRENT_ASSET_CODE, "2026-06-01"]])
    assert MarketContextFetcher._pick_front_future(data, BRENT_ASSET_CODE, "2026-06-25") is None


# ─────────────── _marketdata_price ───────────────

def test_marketdata_prefers_last():
    data = {"marketdata": {"columns": ["SECID", "LAST", "LCURRENTPRICE"],
                           "data": [["BRN6", 110.5, 110.4]]}}
    assert MarketContextFetcher._marketdata_price(data) == 110.5


def test_marketdata_falls_back_to_lcurrentprice():
    data = {"marketdata": {"columns": ["SECID", "LAST", "LCURRENTPRICE"],
                           "data": [["BRN6", None, 109.9]]}}
    assert MarketContextFetcher._marketdata_price(data) == 109.9


def test_marketdata_empty():
    assert MarketContextFetcher._marketdata_price(None) is None


# ─────────────── graceful fallback при недоступном MOEX ───────────────

def test_fetch_usd_rub_returns_none_without_network(monkeypatch):
    fetcher = MarketContextFetcher(tickers=["SBER"], candles_provider=None, news_aggregator=None)
    monkeypatch.setattr(MarketContextFetcher, "_iss_get", staticmethod(lambda *a, **k: None))
    assert fetcher.fetch_usd_rub() is None


def test_fetch_brent_returns_none_without_network(monkeypatch):
    fetcher = MarketContextFetcher(tickers=["SBER"], candles_provider=None, news_aggregator=None)
    monkeypatch.setattr(MarketContextFetcher, "_iss_get", staticmethod(lambda *a, **k: None))
    assert fetcher.fetch_brent_usd() is None
