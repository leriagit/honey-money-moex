"""
Технические индикаторы для SignalEngine.

Реализация максимально приближена к классическим определениям,
чтобы значения можно было проверить против TradingView / TA-Lib.

Все функции принимают numpy-массивы или списки float и возвращают
либо массив той же длины (с NaN в head), либо последнее значение.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import math

import numpy as np

from ..schemas import Candle, IndicatorBundle

# SMA window для тренд-индикатора (раньше брался из decision/config.py)
SMA_PERIOD = 20


# ───────────────────────── RSI ──────────────────────────────────


def rsi(prices: Sequence[float], period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (Wilder smoothing).

    Wilder использовал экспоненциальное сглаживание с alpha = 1/period.
    """
    arr = np.asarray(prices, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n <= period:
        return out

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # Первое значение RSI в индексе `period`
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder smoothing для остальных
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


# ───────────────────────── EMA / MACD ───────────────────────────


def ema(prices: Sequence[float], period: int) -> np.ndarray:
    """Стандартный EMA с alpha = 2 / (period + 1)."""
    arr = np.asarray(prices, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    # Seed первой точкой — простой, но устойчивый вариант
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd(
    prices: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (macd_line, signal_line, hist).
    macd_line = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    """
    arr = np.asarray(prices, dtype=float)
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def macd_cross(macd_line: np.ndarray, signal_line: np.ndarray) -> Tuple[bool, bool]:
    """
    Проверяет пересечение MACD и сигнальной линии на последнем баре.
    Возвращает (cross_up, cross_down).
    """
    if len(macd_line) < 2 or len(signal_line) < 2:
        return False, False
    prev_diff = macd_line[-2] - signal_line[-2]
    curr_diff = macd_line[-1] - signal_line[-1]
    if np.isnan(prev_diff) or np.isnan(curr_diff):
        return False, False
    cross_up = bool(prev_diff <= 0 and curr_diff > 0)
    cross_down = bool(prev_diff >= 0 and curr_diff < 0)
    return cross_up, cross_down


# ───────────────────────── SMA ──────────────────────────────────


def sma(prices: Sequence[float], period: int) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    csum = np.cumsum(arr)
    out[period - 1] = csum[period - 1] / period
    out[period:] = (csum[period:] - csum[:-period]) / period
    return out


# ───────────────────────── Volume helpers ───────────────────────


def volume_ratio(volumes: Sequence[float], window: int = 20) -> Optional[float]:
    """Отношение последнего объёма к среднему за `window`."""
    arr = np.asarray(volumes, dtype=float)
    if len(arr) < window + 1:
        return None
    avg = arr[-window - 1:-1].mean()
    if avg <= 0:
        return None
    return float(arr[-1] / avg)


def volume_avg(volumes: Sequence[float], window: int = 20) -> Optional[float]:
    arr = np.asarray(volumes, dtype=float)
    if len(arr) < window:
        return None
    return float(arr[-window:].mean())


# ───────────────────────── Сборка bundle ────────────────────────


def compute_indicators(candles: List[Candle]) -> IndicatorBundle:
    """
    Главная точка входа: берёт список свечей и считает все нужные
    индикаторы для SignalEngine за один проход.

    Если данных мало — возвращает bundle с теми полями, которые удалось посчитать.
    """
    if not candles:
        return IndicatorBundle()

    closes = np.array([c.close for c in candles], dtype=float)
    volumes = np.array([c.volume for c in candles], dtype=float)
    price = float(closes[-1])

    bundle = IndicatorBundle(price=price)

    # RSI
    if len(closes) >= 15:
        rsi_arr = rsi(closes, period=14)
        last = rsi_arr[-1]
        if not math.isnan(last):
            bundle.rsi_14 = float(last)

    # MACD
    if len(closes) >= 35:
        macd_line, signal_line, hist = macd(closes)
        if not math.isnan(macd_line[-1]) and not math.isnan(signal_line[-1]):
            bundle.macd = float(macd_line[-1])
            bundle.macd_signal = float(signal_line[-1])
            bundle.macd_hist = float(hist[-1])
            up, down = macd_cross(macd_line, signal_line)
            bundle.macd_cross_up = up
            bundle.macd_cross_down = down

    # SMA 20
    if len(closes) >= SMA_PERIOD:
        sma_arr = sma(closes, SMA_PERIOD)
        if not math.isnan(sma_arr[-1]):
            bundle.sma_20 = float(sma_arr[-1])

    # Volume
    bundle.volume_last = float(volumes[-1]) if len(volumes) else None
    bundle.volume_avg_20 = volume_avg(volumes, window=20)
    bundle.volume_ratio = volume_ratio(volumes, window=20)

    return bundle
