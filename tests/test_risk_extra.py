"""Тесты подключённых модулей risk_extra: CrisisGate, StopLossManager, kelly."""
from __future__ import annotations

import tempfile

import pytest

from src.risk_extra.crisis_gate import CrisisGate, CrisisState
from src.risk_extra.position_sizer import kelly_multiplier, signed_kelly
from src.risk_extra.stop_loss import ExitReason, StopLossManager


# ─────────────── CrisisGate ───────────────

def test_crisis_gate_triggers_on_crash():
    cg = CrisisGate(state_path=tempfile.mktemp(suffix=".json"))
    assert cg.update([100.0, 99.0]) == CrisisState.NORMAL      # -1% — норма
    assert cg.update([100.0, 97.0]) == CrisisState.CRISIS      # -3% — кризис
    assert cg.should_force_sell_longs()
    assert cg.is_buy_blocked()


def test_crisis_gate_ignores_short_series():
    cg = CrisisGate(state_path=tempfile.mktemp(suffix=".json"))
    assert cg.update([100.0]) == CrisisState.NORMAL


# ─────────────── StopLossManager ───────────────

@pytest.fixture
def slm() -> StopLossManager:
    m = StopLossManager(state_path=tempfile.mktemp(suffix=".json"))
    m.on_buy("SBER", qty=10, price=100.0, ts="2026-05-25T10:00:00")
    return m


def test_stop_loss_full_exit(slm):
    d = slm.check_position("SBER", 97.0, now_ts=1000.0)   # -3% < -2.4%
    assert d.should_exit and d.fraction == 1.0
    assert d.reason == ExitReason.STOP_LOSS


def test_take_profit_half(slm):
    d = slm.check_position("SBER", 102.0, now_ts=1000.0)  # +2% > +1.5%
    assert d.should_exit and d.fraction == 0.5
    assert d.reason == ExitReason.TAKE_PROFIT_HALF


def test_no_exit_in_band(slm):
    assert not slm.check_position("SBER", 100.5, now_ts=1000.0).should_exit


def test_buy_blocked_after_stop(slm):
    slm.check_position("SBER", 97.0, now_ts=1000.0)         # триггерим стоп
    assert slm.is_buy_blocked("SBER", now_ts=1100.0)        # внутри cooldown
    assert not slm.is_buy_blocked("SBER", now_ts=1000.0 + 601)  # после cooldown


# ─────────────── kelly position sizer ───────────────

@pytest.mark.parametrize("prob,expected", [
    (0.5, 0.7),   # нет edge → base
    (0.7, 1.1),
    (1.0, 1.5),   # cap
])
def test_kelly_multiplier(prob, expected):
    assert kelly_multiplier(prob) == pytest.approx(expected, abs=1e-6)


def test_signed_kelly_direction():
    assert signed_kelly(0.7) > 0    # long
    assert signed_kelly(0.3) < 0    # short
    assert signed_kelly(0.5) == 0
