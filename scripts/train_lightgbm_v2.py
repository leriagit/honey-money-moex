"""train_lightgbm_v2 — улучшенное обучение модели honey_money.

Отличия от train_lightgbm.py (адресуют слабый AUC и overfitting боевой модели):
  1. ТАРГЕТ triple-barrier вместо «вырастет ли через 1 час» — метка по первому
     сработавшему барьеру (тейк/стоп), согласована со стратегией бота.
  2. УСИЛЕННАЯ РЕГУЛЯРИЗАЦИЯ (мельче деревья, больше min_data_in_leaf и L1/L2) —
     сокращает train/test gap (0.61 train → 0.56 test у v1).
  3. EMBARGO на границах сплита — убирает «протекание» меток с горизонтом между
     train/val/test (иначе val/test оптимистичны).
  4. ПОЛНЫЙ НАБОР МЕТРИК через src.eval: AUC/PR-AUC/Brier/калибровка, дециль-lift,
     precision@confidence и сигнальный бэктест (Sharpe/DD/hit-rate), а не только AUC.

Требует доступ к данным MOEX ISS (aiomoex). Запуск:
  python -m scripts.train_lightgbm_v2 --tp 0.015 --sl 0.024 --horizon 8 \
      --out models/lightgbm_v2.txt
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.eval.labeling import triple_barrier_labels
from src.eval.metrics import full_report

logger = logging.getLogger("train_v2")

# Регуляризованные параметры (против overfitting v1: num_leaves 127→31 и т.д.)
LGB_PARAMS_V2 = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "learning_rate": 0.02,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.0,
    "seed": 42,
    "deterministic": True,
    "verbose": -1,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 120


def build_dataset_tb(
    candles_by_ticker: Dict[str, pd.DataFrame],
    macro: pd.DataFrame,
    tp: float, sl: float, horizon: int,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, List[str]]:
    """Фичи (как у бота) + triple-barrier таргет + fwd_ret для бэктеста."""
    from src.data.features import build_feature_matrix, TICKER_TO_CODE

    rows = []
    for ticker, df in candles_by_ticker.items():
        if ticker not in TICKER_TO_CODE:
            continue
        feats = build_feature_matrix(df, macro, ticker, regime_cols=True)
        if feats.empty:
            continue
        idx = feats.index
        c = df.set_index("ts")["close"].reindex(idx)
        h = df.set_index("ts")["high"].reindex(idx) if "high" in df else c
        low = df.set_index("ts")["low"].reindex(idx) if "low" in df else c
        labels, fwd = triple_barrier_labels(c.values, h.values, low.values, tp, sl, horizon)
        feats = feats.copy()
        feats["__target__"] = labels
        feats["__fwd__"] = fwd
        feats["__ticker__"] = ticker
        feats["__ts__"] = idx
        rows.append(feats[labels != -1])

    full = pd.concat(rows, axis=0, ignore_index=True)
    full = full.dropna(subset=["__target__"])
    full = full.sort_values("__ts__").reset_index(drop=True)
    y = full["__target__"].astype("int8")
    fwd = full["__fwd__"].to_numpy(dtype=float)
    feat_cols = [c for c in full.columns if c not in ("__target__", "__fwd__", "__ticker__", "__ts__")]
    return full[feat_cols], y, fwd, feat_cols


def split_with_embargo(n: int, val_frac=0.15, test_frac=0.15, embargo: int = 8):
    """Хронологический split с embargo-зазором между блоками."""
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    tr = np.arange(0, max(0, n_train - embargo))
    va = np.arange(n_train, max(n_train, n_train + n_val - embargo))
    te = np.arange(n_train + n_val, n)
    return tr, va, te


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-05-23")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--tp", type=float, default=0.015)
    ap.add_argument("--sl", type=float, default=0.024)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="models/lightgbm_v2.txt")
    args = ap.parse_args()

    import lightgbm as lgb
    from scripts.train_lightgbm import fetch_candles, build_macro_frame, DEFAULT_TICKERS

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS
    logger.info("1/5 загрузка свечей %d тикеров", len(tickers))
    candles = fetch_candles(tickers, args.start, args.end, args.interval)
    logger.info("2/5 макро-фрейм")
    macro = build_macro_frame(args.start, args.end)
    logger.info("3/5 фичи + triple-barrier (tp=%.3f sl=%.3f h=%d)", args.tp, args.sl, args.horizon)
    X, y, fwd, feat_cols = build_dataset_tb(candles, macro, args.tp, args.sl, args.horizon)
    logger.info("   датасет: %d строк, %d фичей, base_rate=%.3f", len(X), len(feat_cols), y.mean())

    tr, va, te = split_with_embargo(len(X), embargo=args.horizon)
    dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
    dva = lgb.Dataset(X.iloc[va], label=y.iloc[va])
    logger.info("4/5 обучение v2 (регуляризовано)")
    booster = lgb.train(
        LGB_PARAMS_V2, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(100)],
    )

    logger.info("5/5 метрики (полный набор) на тесте")
    ptr = booster.predict(X.iloc[tr])
    pte = booster.predict(X.iloc[te])
    report = full_report(y.iloc[te].to_numpy(), pte, fwd_ret=fwd[te])
    from src.eval.metrics import classification_metrics
    report["train_auc"] = classification_metrics(y.iloc[tr].to_numpy(), ptr)["roc_auc"]
    report["meta"] = {
        "tp": args.tp, "sl": args.sl, "horizon": args.horizon,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "feature_columns": feat_cols, "params": LGB_PARAMS_V2,
    }

    booster.save_model(args.out)
    with open(args.out.replace(".txt", "_metrics.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    c = report["classification"]
    bt = report["backtest"]
    logger.info("ГОТОВО: train_auc=%.4f test_auc=%.4f gap=%.4f | top-decile lift=%.2f | "
                "backtest Sharpe=%.2f maxDD=%.1f%% hit=%.3f",
                report["train_auc"], c["roc_auc"], report["train_auc"] - c["roc_auc"],
                report["deciles"][-1]["lift"], bt["sharpe"], bt["max_drawdown"] * 100, bt["hit_rate"])


if __name__ == "__main__":
    main()
