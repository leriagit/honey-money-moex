"""Risk-management модули: stop-loss, crisis gate, position sizing."""
from .stop_loss import StopLossManager, ExitDecision, ExitReason
from .crisis_gate import CrisisGate, CrisisState
from .position_sizer import kelly_multiplier, signed_kelly

__all__ = [
    "StopLossManager",
    "ExitDecision",
    "ExitReason",
    "CrisisGate",
    "CrisisState",
    "kelly_multiplier",
    "signed_kelly",
]
