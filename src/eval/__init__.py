"""Оценка качества торговых ML-сигналов.

Главная идея: ROC-AUC сам по себе плохо отражает прибыльность. Здесь собраны
метрики, которые в нашем случае коррелируют с доходностью портфеля:
  • качество вероятности (AUC, PR-AUC, Brier, калибровка);
  • точность в high-confidence децилях (бот торгует только уверенные сигналы);
  • сигнальный бэктест (Sharpe, max drawdown, hit-rate, оборот) — деньги, а не AUC.
"""
from src.eval.metrics import (
    classification_metrics,
    decile_table,
    calibration_table,
    precision_at_confidence,
    signal_backtest,
    information_coefficient,
    full_report,
)
from src.eval.labeling import triple_barrier_labels

__all__ = [
    "classification_metrics",
    "decile_table",
    "calibration_table",
    "precision_at_confidence",
    "signal_backtest",
    "information_coefficient",
    "full_report",
    "triple_barrier_labels",
]
