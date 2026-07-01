"""Тесты triple-barrier разметки."""
from __future__ import annotations

import numpy as np

from src.eval.labeling import triple_barrier_labels


def test_take_profit_hit_first():
    # цена быстро растёт → должен сработать верхний барьер (метка 1)
    close = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    labels, fwd = triple_barrier_labels(close, tp=0.015, sl=0.024, horizon=4)
    assert labels[0] == 1
    assert fwd[0] > 0


def test_stop_loss_hit_first():
    # цена падает → нижний барьер (метка 0)
    close = np.array([100, 99, 98, 97, 96, 95], dtype=float)
    labels, fwd = triple_barrier_labels(close, tp=0.015, sl=0.024, horizon=4)
    assert labels[0] == 0
    assert fwd[0] < 0


def test_timeout_uses_sign():
    # боковик в пределах барьеров → метка по знаку на горизонте
    close = np.array([100, 100.2, 100.1, 100.3, 100.2], dtype=float)
    labels, fwd = triple_barrier_labels(close, tp=0.05, sl=0.05, horizon=3)
    assert labels[0] in (0, 1)
    assert labels[0] == 1  # close[3] > close[0]


def test_last_bars_marked_invalid():
    close = np.arange(100, 110, dtype=float)
    labels, _ = triple_barrier_labels(close, horizon=4)
    assert labels[-1] == -1  # нет окна вперёд


def test_high_low_intrabar_touch():
    # close спокойный, но high пробивает верхний барьер внутри бара
    close = np.array([100, 100.1, 100.2, 100.1], dtype=float)
    high = np.array([100, 102.0, 100.3, 100.2], dtype=float)
    low = np.array([100, 100.0, 100.1, 100.0], dtype=float)
    labels, fwd = triple_barrier_labels(close, high, low, tp=0.015, sl=0.024, horizon=3)
    assert labels[0] == 1
