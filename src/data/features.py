"""
Feature engineering для LightGBM на MOEX.

Используется подмножество Alpha158 (Qlib) + macro factors + regime flags.
Все фичи вычисляются ТОЛЬКО на основе прошлых баров — no look-ahead bias.

Контракт:
  build_feature_matrix(df_ohlcv, df_macro, ticker, regime_cols=True)
    -> pd.DataFrame с N колонками фич и индексом по `ts`

Входной df_ohlcv (long-form, для одного тикера):
  ts (datetime, UTC), open, high, low, close, volume

Входной df_macro (опционально, daily resolution, ffill до часовых):
  ts, brent_usd, usd_rub, cbr_rate, gold_usd, imoex_close

Колонки фич:
  ── Returns (8) ───────────
    ret_1, ret_5, ret_10, ret_20, ret_60
    log_ret_1, log_ret_5, log_ret_20
  ── Volatility (5) ─────────
    vol_5, vol_10, vol_20, vol_60, ret_skew_20
  ── Moving Averages (8) ────
    sma5_ratio, sma10_ratio, sma20_ratio, sma60_ratio
    ema12_ratio, ema26_ratio, sma20_slope, ema12_minus_ema26
  ── Momentum (6) ───────────
    rsi_14, macd_line, macd_signal, macd_hist
    momentum_5, momentum_20
  ── Volume (6) ─────────────
    vol_ratio_5, vol_ratio_20, vol_zscore_20
    obv_change_5, log_volume, volume_imbalance_5
  ── Range / Bands (5) ──────
    high_low_range_5, high_low_range_20
    bband_position, atr_14_ratio, gap_open_close
  ── Time (4) ───────────────
    hour_sin, hour_cos, dow_sin, dow_cos
  ── Macro (5, joined daily) ─
    brent_log, usd_rub_log, cbr_rate_pct
    gold_log, imoex_ret_5
  ── Regime (2) ─────────────
    is_post_feb2022, days_since_regime_change
  ── Ticker (1, categorical, отдельно) ─
    ticker_code (через ordinal encoding, передаётся как cat_feature в LGB)

Итого: ~50 фич + 1 категориальный ticker_code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Дата начала post-2022 режима (примерно момент санкций / закрытия торгов март 2022)
REGIME_CHANGE_DATE = pd.Timestamp("2022-03-24", tz="UTC")

# Стандартный порядок тикеров — фиксированный для воспроизводимости encoding'а.
# Точный список 19 активов из ТЗ ArenaGo (см. docs/COMPETITION.md):
#   Банки/Финансы: SBER, VTBR, T (TCS), MOEX
#   Нефтегаз:       LKOH, ROSN, GAZP, NVTK, SNGSP
#   Металлы/Майнинг: GMKN, ALRS, CHMF, NLMK, PLZL
#   IT:             YDEX
#   Транспорт:      AFLT
#   Ритейл:         X5, MGNT
#   Девелопмент:    PIKK
#   Телеком:        MTSS
TICKER_ORDER = [
    "SBER", "VTBR", "T", "MOEX",
    "LKOH", "ROSN", "GAZP", "NVTK", "SNGSP",
    "GMKN", "ALRS", "CHMF", "NLMK", "PLZL",
    "YDEX",
    "AFLT",
    "X5", "MGNT",
    "PIKK",
    "MTSS",
]
TICKER_TO_CODE: Dict[str, int] = {t: i for i, t in enumerate(TICKER_ORDER)}


# ────────────────────── Базовые числовые утилиты ──────────────────


def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def _safe_log_ret(s: pd.Series, periods: int = 1) -> pd.Series:
    shifted = s.shift(periods)
    return np.log(s / shifted).replace([np.inf, -np.inf], np.nan)


def _rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.pct_change().rolling(window).std()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - 100 / (1 + rs)
    return rsi_val.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_sig = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


# ────────────────────── Главная функция ──────────────────────


@dataclass
class FeatureConfig:
    """Конфиг для build_feature_matrix."""
    include_macro: bool = True
    include_regime: bool = True
    include_ticker_code: bool = True
    dropna_target_rows: bool = True
    target_horizon: int = 1   # за сколько баров вперёд предсказываем return
    target_threshold: float = 0.0  # 0.0 → бинарный sign(return)


def build_feature_matrix(
    df_ohlcv: pd.DataFrame,
    df_macro: Optional[pd.DataFrame] = None,
    ticker: str = "UNKN",
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Строит матрицу фич для одного тикера. Возвращает DataFrame с колонками
    фич + target + ts (как index). Включает только строки, где target определён.

    Все фичи вычисляются без look-ahead bias (только прошлые бары).
    """
    cfg = config or FeatureConfig()
    df = df_ohlcv.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # Базовые серии
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    feat: Dict[str, pd.Series] = {}

    # ── Returns ──
    feat["ret_1"] = _safe_pct_change(c, 1)
    feat["ret_5"] = _safe_pct_change(c, 5)
    feat["ret_10"] = _safe_pct_change(c, 10)
    feat["ret_20"] = _safe_pct_change(c, 20)
    feat["ret_60"] = _safe_pct_change(c, 60)
    feat["log_ret_1"] = _safe_log_ret(c, 1)
    feat["log_ret_5"] = _safe_log_ret(c, 5)
    feat["log_ret_20"] = _safe_log_ret(c, 20)

    # ── Volatility ──
    feat["vol_5"] = _rolling_std(c, 5)
    feat["vol_10"] = _rolling_std(c, 10)
    feat["vol_20"] = _rolling_std(c, 20)
    feat["vol_60"] = _rolling_std(c, 60)
    feat["ret_skew_20"] = feat["ret_1"].rolling(20).skew()

    # ── Moving averages (ratios so they're stationary) ──
    sma5 = c.rolling(5).mean()
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    sma60 = c.rolling(60).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    feat["sma5_ratio"] = (c / sma5) - 1.0
    feat["sma10_ratio"] = (c / sma10) - 1.0
    feat["sma20_ratio"] = (c / sma20) - 1.0
    feat["sma60_ratio"] = (c / sma60) - 1.0
    feat["ema12_ratio"] = (c / ema12) - 1.0
    feat["ema26_ratio"] = (c / ema26) - 1.0
    feat["sma20_slope"] = (sma20 - sma20.shift(5)) / sma20.shift(5)
    feat["ema12_minus_ema26"] = (ema12 - ema26) / c

    # ── Momentum / Oscillators ──
    feat["rsi_14"] = (_rsi(c, 14) - 50.0) / 50.0   # центрируем в [-1, 1]
    macd_line, macd_sig, macd_hist = _macd(c)
    feat["macd_line"] = macd_line / c
    feat["macd_signal"] = macd_sig / c
    feat["macd_hist"] = macd_hist / c
    feat["momentum_5"] = c / c.shift(5) - 1.0
    feat["momentum_20"] = c / c.shift(20) - 1.0

    # ── Volume ──
    v_safe = v.replace(0, np.nan)
    feat["vol_ratio_5"] = v_safe / v_safe.rolling(5).mean()
    feat["vol_ratio_20"] = v_safe / v_safe.rolling(20).mean()
    vol_std_20 = v_safe.rolling(20).std()
    feat["vol_zscore_20"] = (v_safe - v_safe.rolling(20).mean()) / vol_std_20
    obv = _obv(c, v)
    feat["obv_change_5"] = obv.diff(5) / v_safe.rolling(20).mean()
    feat["log_volume"] = np.log1p(v_safe)
    # imbalance: чистый знаковый объём за 5 баров / общий объём
    direction = np.sign(c.diff().fillna(0.0))
    signed_vol = direction * v_safe
    feat["volume_imbalance_5"] = signed_vol.rolling(5).sum() / v_safe.rolling(5).sum().replace(0, np.nan)

    # ── Range / Bands ──
    feat["high_low_range_5"] = (h - l).rolling(5).mean() / c
    feat["high_low_range_20"] = (h - l).rolling(20).mean() / c
    bb_mid = sma20
    bb_std = c.rolling(20).std()
    feat["bband_position"] = (c - bb_mid) / (2.0 * bb_std.replace(0, np.nan))
    atr14 = _atr(h, l, c, 14)
    feat["atr_14_ratio"] = atr14 / c
    feat["gap_open_close"] = (o - c.shift(1)) / c.shift(1)

    # ── Time of day / day of week (cyclical encoding) ──
    ts = pd.to_datetime(df["ts"], utc=True)
    hours = ts.dt.hour + ts.dt.minute / 60.0
    feat["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    feat["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    feat["dow_sin"] = np.sin(2.0 * np.pi * ts.dt.dayofweek / 7.0)
    feat["dow_cos"] = np.cos(2.0 * np.pi * ts.dt.dayofweek / 7.0)

    # ── Macro features (join по ts → forward-fill) ──
    if cfg.include_macro and df_macro is not None and not df_macro.empty:
        m = df_macro.copy().sort_values("ts")
        m["ts"] = pd.to_datetime(m["ts"], utc=True)
        # join asof по ts — берём самое свежее доступное значение макро
        merged = pd.merge_asof(
            pd.DataFrame({"ts": ts, "_idx": np.arange(len(ts))}).sort_values("ts"),
            m,
            on="ts",
            direction="backward",
        ).sort_values("_idx").reset_index(drop=True)

        if "brent_usd" in merged.columns:
            feat["brent_log"] = np.log(merged["brent_usd"].clip(lower=1))
        if "usd_rub" in merged.columns:
            feat["usd_rub_log"] = np.log(merged["usd_rub"].clip(lower=1))
        if "cbr_rate" in merged.columns:
            feat["cbr_rate_pct"] = merged["cbr_rate"] / 100.0
        if "gold_usd" in merged.columns:
            feat["gold_log"] = np.log(merged["gold_usd"].clip(lower=1))
        if "imoex_close" in merged.columns:
            feat["imoex_ret_5"] = merged["imoex_close"].pct_change(5)

    # ── Regime ──
    if cfg.include_regime:
        feat["is_post_feb2022"] = (ts >= REGIME_CHANGE_DATE).astype(float)
        days_since = (ts - REGIME_CHANGE_DATE).dt.total_seconds() / 86400.0
        feat["days_since_regime_change"] = days_since.clip(lower=-365 * 5, upper=365 * 5) / 365.0

    # ── Ticker code (categorical) ──
    if cfg.include_ticker_code:
        feat["ticker_code"] = TICKER_TO_CODE.get(ticker, len(TICKER_ORDER))

    # ── Target ──
    target_return = c.shift(-cfg.target_horizon) / c - 1.0
    target = (target_return > cfg.target_threshold).astype(float)
    feat["target_return"] = target_return
    feat["target"] = target

    # Собираем DataFrame
    out = pd.DataFrame(feat)
    out.index = ts
    out["ticker"] = ticker

    # Чистим первые/последние строки от NaN от роллингов и таргета
    feature_cols = [
        col for col in out.columns
        if col not in ("target", "target_return", "ticker")
    ]
    # Дропаем строки, где >50% фич NaN (для самых первых баров)
    out = out.dropna(subset=feature_cols, thresh=int(0.5 * len(feature_cols)))
    if cfg.dropna_target_rows:
        out = out.dropna(subset=["target"])

    return out


def get_feature_column_names(
    include_macro: bool = True,
    include_regime: bool = True,
    include_ticker_code: bool = True,
) -> List[str]:
    """
    Канонический список имён фич — должен совпадать с теми, что строит
    build_feature_matrix. Используется при инференсе для отбора колонок.
    """
    cols = [
        # returns
        "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
        "log_ret_1", "log_ret_5", "log_ret_20",
        # vol
        "vol_5", "vol_10", "vol_20", "vol_60", "ret_skew_20",
        # MA
        "sma5_ratio", "sma10_ratio", "sma20_ratio", "sma60_ratio",
        "ema12_ratio", "ema26_ratio", "sma20_slope", "ema12_minus_ema26",
        # Momentum
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "momentum_5", "momentum_20",
        # Volume
        "vol_ratio_5", "vol_ratio_20", "vol_zscore_20",
        "obv_change_5", "log_volume", "volume_imbalance_5",
        # Range
        "high_low_range_5", "high_low_range_20",
        "bband_position", "atr_14_ratio", "gap_open_close",
        # Time
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    if include_macro:
        cols += ["brent_log", "usd_rub_log", "cbr_rate_pct", "gold_log", "imoex_ret_5"]
    if include_regime:
        cols += ["is_post_feb2022", "days_since_regime_change"]
    if include_ticker_code:
        cols += ["ticker_code"]
    return cols


# ─────────────── Online featurize для инференса ───────────────────


def featurize_single_window(
    candles_df: pd.DataFrame,
    macro_snapshot: Optional[Dict[str, float]],
    ticker: str,
    now_ts: Optional[pd.Timestamp] = None,
    config: Optional[FeatureConfig] = None,
) -> Optional[pd.Series]:
    """
    Считает фичи на одной точке (последний бар = `now_ts`) для инференса
    в LightGBMMLProvider.predict().

    Возвращает Series с именами колонок == get_feature_column_names().
    Если данных недостаточно (< 60 баров) — возвращает None.
    """
    if candles_df is None or len(candles_df) < 60:
        return None

    cfg = config or FeatureConfig(dropna_target_rows=False)
    # macro_snapshot → df_macro единичный
    df_macro = None
    if macro_snapshot:
        df_macro = pd.DataFrame([{
            "ts": pd.to_datetime(candles_df["ts"].iloc[0], utc=True),
            **macro_snapshot,
        }])

    full = build_feature_matrix(
        candles_df,
        df_macro=df_macro,
        ticker=ticker,
        config=cfg,
    )
    if full.empty:
        return None
    # берём последнюю валидную строку
    last = full.iloc[-1]
    feature_cols = get_feature_column_names(
        include_macro=cfg.include_macro,
        include_regime=cfg.include_regime,
        include_ticker_code=cfg.include_ticker_code,
    )
    # отдаём только фичи (без target/ticker)
    available = [c for c in feature_cols if c in last.index]
    return last[available]
