"""backtest_signal — сигнальный бэктест обученной модели на данных MOEX.

Берёт сохранённую LightGBM-модель, прогоняет её по часовым свечам тикеров,
строит prob_up на каждом баре и считает деньги-метрики стратегии:
совокупная доходность, Sharpe, max drawdown, hit-rate, оборот.

Результат печатается и пишется в JSON — это реальные числа для слайда
«Результаты» и для дашборда (вместо текущих заглушек).

Запуск:
  python -m scripts.backtest_signal --model models/lightgbm_v2.txt \
      --long-thr 0.56 --short-thr 0.44 --out backtest_result.json
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src.eval.metrics import signal_backtest, precision_at_confidence, classification_metrics

logger = logging.getLogger("backtest")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/lightgbm_v2.txt")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-05-23")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--long-thr", type=float, default=0.56)
    ap.add_argument("--short-thr", type=float, default=0.44)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--out", default="backtest_result.json")
    args = ap.parse_args()

    import lightgbm as lgb
    from scripts.train_lightgbm import fetch_candles, build_macro_frame, DEFAULT_TICKERS
    from src.data.features import build_feature_matrix, TICKER_TO_CODE

    booster = lgb.Booster(model_file=args.model)
    feat_cols = None
    try:
        meta = json.load(open(args.model.replace(".txt", "_metrics.json")))
        feat_cols = meta.get("meta", {}).get("feature_columns")
    except Exception:
        pass

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS
    candles = fetch_candles(tickers, args.start, args.end, args.interval)
    macro = build_macro_frame(args.start, args.end)

    all_prob, all_fwd, all_y = [], [], []
    per_ticker = {}
    for ticker, df in candles.items():
        if ticker not in TICKER_TO_CODE:
            continue
        feats = build_feature_matrix(df, macro, ticker, regime_cols=True)
        if feats.empty:
            continue
        cols = feat_cols or [c for c in feats.columns]
        X = feats.reindex(columns=cols)
        prob = booster.predict(X)
        close = df.set_index("ts")["close"].reindex(feats.index)
        fwd = (close.shift(-1) / close - 1.0).to_numpy()  # доходность следующего бара
        mask = ~np.isnan(fwd)
        prob, fwd = np.asarray(prob)[mask], fwd[mask]
        y = (fwd > 0).astype(int)
        bt = signal_backtest(prob, fwd, args.long_thr, args.short_thr, args.cost)
        per_ticker[ticker] = {"auc": classification_metrics(y, prob)["roc_auc"], **{k: bt[k] for k in ("sharpe", "total_return", "hit_rate")}}
        all_prob.append(prob); all_fwd.append(fwd); all_y.append(y)

    prob = np.concatenate(all_prob); fwd = np.concatenate(all_fwd); y = np.concatenate(all_y)
    overall = {
        "classification": classification_metrics(y, prob),
        "precision@0.60": precision_at_confidence(y, prob, 0.60),
        "backtest": signal_backtest(prob, fwd, args.long_thr, args.short_thr, args.cost),
        "per_ticker": per_ticker,
        "params": {"long_thr": args.long_thr, "short_thr": args.short_thr, "cost": args.cost},
    }
    with open(args.out, "w") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    bt = overall["backtest"]
    logger.info("РЕЗУЛЬТАТ: AUC=%.4f | доходность=%.2f%% Sharpe=%.2f maxDD=%.1f%% hit=%.3f сделок=%d",
                overall["classification"]["roc_auc"], bt["total_return"] * 100, bt["sharpe"],
                bt["max_drawdown"] * 100, bt["hit_rate"], bt["n_trades"])
    logger.info("записано в %s", args.out)


if __name__ == "__main__":
    main()
