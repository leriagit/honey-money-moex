"""Расчёт количественных сигналов по свечам.

Выделено из honey_money_provider.py. Отвечает за две вещи:
  • технические индикаторы (RSI / MACD / volume_ratio / SMA20-ratio);
  • ML-вероятность роста (LightGBM × MacroContext) и ожидаемая волатильность.

Конвертация aiomoex-словарей в объекты-свечи тоже здесь, чтобы провайдер
не знал о форматах данных.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Минимум баров, ниже которого сигнал не считаем (индикаторам нужна история).
_MIN_BARS_INDICATORS = 30
_MIN_BARS_ML = 60


class SignalComputer:
    """Считает технические индикаторы и ML-сигналы по тикерам."""

    def __init__(self, ml_provider: Optional[Any]) -> None:
        self._ml_provider = ml_provider
        self.expected_vols: Dict[str, float] = {}

    @staticmethod
    def to_candle_objects(raw: list) -> list:
        """Конвертирует list[dict] из aiomoex в list[Candle-сурогат] для индикаторов."""
        out = []
        for r in raw:
            ts_str = r.get("begin") or r.get("ts")
            if isinstance(ts_str, str):
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.now(timezone.utc)
            else:
                ts = ts_str or datetime.now(timezone.utc)
            out.append(SimpleNamespace(
                ts=ts,
                open=float(r.get("open", 0)),
                high=float(r.get("high", 0)),
                low=float(r.get("low", 0)),
                close=float(r.get("close", 0)),
                volume=float(r.get("volume", 0) or 0),
            ))
        return out

    def compute_indicators(
        self, candles_by_ticker: Dict[str, list],
    ) -> Dict[str, Dict[str, Any]]:
        """Для каждого тикера: RSI, MACD-hist, volume_ratio, SMA20-ratio."""
        out: Dict[str, Dict[str, Any]] = {}
        try:
            from src.data.indicators import compute_indicators as _ci
        except ImportError as e:
            logger.warning("compute_indicators import failed: %s", e)
            return out

        for ticker, raw_candles in candles_by_ticker.items():
            if not raw_candles or len(raw_candles) < _MIN_BARS_INDICATORS:
                out[ticker] = {}
                continue
            try:
                ind = _ci(self.to_candle_objects(raw_candles))
                out[ticker] = {
                    "rsi_14": ind.rsi_14,
                    "macd_hist": ind.macd_hist,
                    "volume_ratio": ind.volume_ratio,
                    "sma20_ratio": (ind.price / ind.sma_20 - 1.0)
                    if (ind.sma_20 and ind.price) else 0.0,
                }
            except Exception as e:
                logger.warning("Indicators for %s failed: %s", ticker, e)
                out[ticker] = {}
        return out

    def compute_ml_signals(
        self, candles_by_ticker: Dict[str, list],
    ) -> Dict[str, float]:
        """Возвращает {ticker: prob_up}. Параллельно копит expected_vols для сайзинга."""
        self.expected_vols = {}
        if self._ml_provider is None:
            return {}

        out: Dict[str, float] = {}
        for ticker, raw_candles in candles_by_ticker.items():
            if not raw_candles or len(raw_candles) < _MIN_BARS_ML:
                continue
            try:
                candles = self.to_candle_objects(raw_candles)
                sig = self._ml_provider.predict(candles, news=None, horizon="1h", ticker=ticker)
                out[ticker] = float(sig.prob_up)
                if sig.expected_vol is not None:
                    self.expected_vols[ticker] = float(sig.expected_vol)
            except Exception as e:
                logger.warning("ML predict for %s failed: %s", ticker, e)
        return out
