"""
Kelly-подобный position sizing.

Идея: модель возвращает prob_up ∈ [0, 1]. Уверенность модели = |prob_up - 0.5| × 2.
  - prob_up=0.5 (полная неуверенность) → 0% уверенности
  - prob_up=0.6 → 20% уверенности
  - prob_up=0.7 → 40% уверенности
  - prob_up=0.8 → 60% уверенности

Размер позиции = base × multiplier(edge):
  multiplier = clip(0.3 + edge × 1.4, min=0.3, max=1.5)

То есть:
  prob_up=0.5 (нет edge) → 30% от base (минимум, чтобы делать сделку для оборота)
  prob_up=0.6 → 58% от base
  prob_up=0.7 → 86% от base
  prob_up=0.8 → 114% от base (максимум)
  prob_up=0.9 → 142% от base
  prob_up=1.0 → 150% от base (cap)

В backtest и production используем как множитель к size_fraction из SignalEngine.

Это "soft Kelly" — без полной формулы Kelly (которая требует знания
expected return и variance). У нас есть только prob_up — этого достаточно
для качественной подстройки размера.
"""
from __future__ import annotations


def kelly_multiplier(
    prob_up: float,
    min_mult: float = 0.7,   # ⬆ было 0.3 — слишком жёстко резало оборот
    max_mult: float = 1.5,
    base_mult: float = 0.7,  # ⬆ было 0.3
    edge_scale: float = 1.0, # ⬇ было 1.4
) -> float:
    """
    Возвращает множитель к размеру позиции (0.3 .. 1.5).

    edge = |prob_up - 0.5| × 2 ∈ [0, 1]
    multiplier = clip(base_mult + edge × edge_scale, min_mult, max_mult)
    """
    if prob_up < 0 or prob_up > 1:
        return 1.0  # safe default
    edge = abs(prob_up - 0.5) * 2.0
    mult = base_mult + edge * edge_scale
    return max(min_mult, min(max_mult, mult))


def signed_kelly(prob_up: float, max_mult: float = 1.5) -> float:
    """
    Версия со знаком: положительная = LONG, отрицательная = SHORT.

    prob_up=0.5 → 0
    prob_up=0.7 → +max_mult × 0.4 = +0.6 (long)
    prob_up=0.3 → -max_mult × 0.4 = -0.6 (short)
    """
    edge = (prob_up - 0.5) * 2.0  # [-1, +1]
    return edge * max_mult
