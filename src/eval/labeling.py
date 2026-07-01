"""Triple-barrier разметка таргета (López de Prado).

Зачем: исходная модель учится на «вырастет ли close через 1 час» — это почти
монетка (AUC ≈ 0.56). Triple-barrier ставит метку по тому, что РЕАЛЬНО приносит
деньги: какой барьер сработал первым — тейк (+tp) или стоп (−sl) — в окне horizon.
Это согласует ML-цель со стратегией mean-reversion бота (stop 2.4% / take 1.5%).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def triple_barrier_labels(
    close, high=None, low=None,
    tp: float = 0.015, sl: float = 0.024, horizon: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Размечает каждый бар по первому сработавшему барьеру.

    Метка: 1 если первым достигнут верхний барьер close·(1+tp),
           0 если первым нижний close·(1−sl),
           если ни один за horizon баров — по знаку доходности на горизонте.
    Барьеры проверяются по high/low (внутрибарно); если high/low не заданы —
    используется close.

    Возвращает (labels[int8], fwd_ret[float]) длины как у close.
    Последние бары без полного окна получают метку −1 (их нужно отбросить).
    """
    close = np.asarray(close, dtype=float).ravel()
    high = close if high is None else np.asarray(high, dtype=float).ravel()
    low = close if low is None else np.asarray(low, dtype=float).ravel()
    n = close.size
    labels = np.full(n, -1, dtype=np.int8)
    fwd = np.zeros(n, dtype=float)

    for i in range(n):
        end = min(i + horizon, n - 1)
        if end <= i:
            break  # дальше окна не хватает
        up = close[i] * (1.0 + tp)
        dn = close[i] * (1.0 - sl)
        decided = False
        for j in range(i + 1, end + 1):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn
            if hit_up and hit_dn:
                labels[i] = 1 if close[j] >= close[i] else 0
                fwd[i] = close[j] / close[i] - 1.0
                decided = True
                break
            if hit_up:
                labels[i] = 1
                fwd[i] = tp
                decided = True
                break
            if hit_dn:
                labels[i] = 0
                fwd[i] = -sl
                decided = True
                break
        if not decided:
            labels[i] = 1 if close[end] > close[i] else 0
            fwd[i] = close[end] / close[i] - 1.0
    return labels, fwd
