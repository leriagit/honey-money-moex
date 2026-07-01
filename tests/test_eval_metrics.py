"""Тесты метрик src/eval на синтетике с известными свойствами."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    classification_metrics,
    decile_table,
    calibration_table,
    precision_at_confidence,
    signal_backtest,
    information_coefficient,
    full_report,
)


@pytest.fixture
def perfect():
    # идеальный предиктор: prob = метка
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 50)
    p = y.astype(float) * 0.98 + 0.01
    return y, p


@pytest.fixture
def random_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = rng.uniform(0, 1, 4000)
    return y, p


# ── classification ──

def test_perfect_auc_is_one(perfect):
    m = classification_metrics(*perfect)
    assert m["roc_auc"] == pytest.approx(1.0, abs=1e-6)
    assert m["accuracy"] == pytest.approx(1.0, abs=1e-6)
    assert m["brier"] < 0.01


def test_random_auc_near_half(random_signal):
    m = classification_metrics(*random_signal)
    assert 0.45 < m["roc_auc"] < 0.55


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.5])


# ── deciles ──

def test_decile_monotonic_for_signal():
    # prob с реальным edge → верхний дециль должен иметь больший up-rate
    rng = np.random.default_rng(1)
    n = 5000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)  # факт зависит от prob
    rows = decile_table(y, p, n_bins=10)
    assert len(rows) == 10
    assert rows[-1]["actual_up_rate"] > rows[0]["actual_up_rate"]
    assert rows[-1]["lift"] > 1.0 > rows[0]["lift"]


# ── calibration ──

def test_calibration_low_ece_when_calibrated():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # идеально калибровано
    cal = calibration_table(y, p, bins=10)
    assert cal["ece"] < 0.03


# ── precision@confidence ──

def test_precision_at_confidence(perfect):
    pr = precision_at_confidence(*perfect, thr=0.6)
    assert pr["long_precision"] == pytest.approx(1.0, abs=1e-6)
    assert pr["short_precision"] == pytest.approx(1.0, abs=1e-6)


# ── signal backtest ──

def test_backtest_profitable_when_signal_predicts_return():
    rng = np.random.default_rng(3)
    n = 3000
    fwd = rng.normal(0, 0.01, n)
    # prob коррелирует со знаком будущей доходности → стратегия должна быть в плюс
    p = 0.5 + 0.4 * np.sign(fwd) + rng.normal(0, 0.05, n)
    p = np.clip(p, 0, 1)
    bt = signal_backtest(p, fwd, cost=0.0)
    assert bt["total_return"] > 0
    assert bt["sharpe"] > 0
    assert 0 <= bt["exposure"] <= 1
    assert -1 <= bt["max_drawdown"] <= 0


def test_backtest_costs_reduce_return():
    rng = np.random.default_rng(4)
    n = 2000
    fwd = rng.normal(0, 0.01, n)
    p = np.clip(0.5 + 0.3 * np.sign(fwd) + rng.normal(0, 0.1, n), 0, 1)
    free = signal_backtest(p, fwd, cost=0.0)["total_return"]
    costly = signal_backtest(p, fwd, cost=0.002)["total_return"]
    assert costly < free


def test_backtest_length_guard():
    with pytest.raises(ValueError):
        signal_backtest([0.6, 0.4], [0.01])


def test_risk_adjusted_metrics_present_and_sane():
    rng = np.random.default_rng(5)
    n = 4000
    fwd = rng.normal(0, 0.01, n)
    # умеренный, НЕ идеальный сигнал → есть и выигрыши, и проигрыши (downside существует)
    p = np.clip(0.5 + 0.10 * np.sign(fwd) + rng.normal(0, 0.30, n), 0, 1)
    bt = signal_backtest(p, fwd, cost=0.0)
    for k in ("sortino", "calmar", "profit_factor", "expectancy"):
        assert k in bt
    assert bt["profit_factor"] > 1.0          # прибыльная стратегия
    assert bt["sortino"] > 0 and np.isfinite(bt["sortino"])
    assert bt["calmar"] > 0                    # положительная доходность на просадку


def test_information_coefficient_sign():
    rng = np.random.default_rng(6)
    fwd = rng.normal(0, 0.01, 5000)
    p_good = 0.5 + 0.4 * np.sign(fwd) + rng.normal(0, 0.1, 5000)
    assert information_coefficient(p_good, fwd) > 0.2   # сигнал ранжирует доходность
    assert abs(information_coefficient(rng.uniform(0, 1, 5000), fwd)) < 0.05  # шум → ~0


# ── full report ──

def test_full_report_keys(perfect):
    y, p = perfect
    fwd = (y * 2 - 1) * 0.01
    rep = full_report(y, p, fwd_ret=fwd, cost=0.0)
    assert {"classification", "deciles", "calibration", "backtest"} <= set(rep)
    assert rep["backtest"]["total_return"] > 0
