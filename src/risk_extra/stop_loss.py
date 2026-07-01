"""
StopLossManager — управление выходом из позиций.

Логика (по рекомендациям брифа аналитика):
  - stop-loss: -3% от avg_price → SELL_ALL (полный выход)
  - take-profit: +5% от avg_price → SELL_HALF (фиксируем половину)
  - trailing stop: после достижения +2% движем stop вверх:
    new_stop = max(initial_stop, peak_price × (1 - trailing_pct))

Используется в:
  - backtest_hybrid.py — проверка стопов перед основной decision logic
  - production Executor — то же самое в реальном цикле

Поведение:
  Каждый тик мы вызываем check_position(ticker, price_now, position) и
  получаем ExitDecision. Если decision.should_exit=True — форсируем выход,
  игнорируя сигналы SignalEngine. Это safety net.

State (peak_price для trailing) хранится в /data/stop_loss.json для recovery.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    NONE = "none"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_HALF = "take_profit_half"
    TRAILING_STOP = "trailing_stop"
    HARD_DRAWDOWN = "hard_drawdown"


@dataclass
class ExitDecision:
    """Что делать с позицией. should_exit=False означает не трогать."""
    should_exit: bool
    fraction: float = 0.0           # 0..1 какую долю закрывать
    reason: ExitReason = ExitReason.NONE
    explanation: str = ""


@dataclass
class PositionState:
    """Состояние одной открытой позиции (для trailing stop)."""
    ticker: str
    avg_price: float
    qty: float
    peak_price: float = 0.0  # максимум цены с момента открытия (для trailing)
    opened_ts: Optional[str] = None


class StopLossManager:
    """
    Управляет выходом из позиций. Конфиг по умолчанию основан на
    рекомендациях брифа: stop -3%, take 5%, trailing 2%.
    """

    def __init__(
        self,
        # Параметры -20% (запрос трейдера 27.05) + TP теперь 1.5%:
        stop_loss_pct: float = 0.024,       # -2.4% (было -3%) → force SELL_ALL
        take_profit_pct: float = 0.015,     # +1.5% (было +2%) → SELL_HALF
        trailing_start_pct: float = 0.012,  # активируем trailing после +1.2% (было 1.5%)
        trailing_distance_pct: float = 0.008,  # trailing на 0.8% ниже peak (было 1%)
        hard_drawdown_pct: float = 0.04,    # -4% от peak (было -5%) → force SELL_ALL
        cooldown_seconds: int = 600,        # 10 мин блок BUY после стопа (было 5) — гасит wash trade
        state_path: str = "/data/stop_loss.json",
    ) -> None:
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_start_pct = trailing_start_pct
        self.trailing_distance_pct = trailing_distance_pct
        self.hard_drawdown_pct = hard_drawdown_pct
        self.cooldown_seconds = cooldown_seconds
        self.state_path = Path(state_path)
        self.positions: Dict[str, PositionState] = {}
        self._half_taken: Dict[str, bool] = {}
        # tk → ts (unix-сек) когда последний раз сработал stop. BUY заблокирован
        # до cooldown_seconds после этой ts.
        self._stop_loss_cooldown: Dict[str, float] = {}
        self._load_state()

    # ─────── persistence ───────

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            for tk, p in data.get("positions", {}).items():
                self.positions[tk] = PositionState(**p)
            self._half_taken = dict(data.get("half_taken", {}))
        except Exception as e:
            logger.warning("StopLoss: state load failed: %s", e)

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "positions": {tk: asdict(p) for tk, p in self.positions.items()},
                "half_taken": dict(self._half_taken),
            }
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("StopLoss: state save failed: %s", e)

    # ─────── API позиции ───────

    def on_buy(self, ticker: str, qty: float, price: float, ts: Optional[str] = None) -> None:
        """Обновляем state после BUY (новой или докупки)."""
        existing = self.positions.get(ticker)
        if existing is None or existing.qty <= 0:
            # новая позиция
            self.positions[ticker] = PositionState(
                ticker=ticker, avg_price=price, qty=qty, peak_price=price, opened_ts=ts,
            )
            self._half_taken[ticker] = False
        else:
            # докупка — усредняем avg_price, peak обновляем
            new_qty = existing.qty + qty
            new_avg = (existing.avg_price * existing.qty + price * qty) / new_qty
            existing.qty = new_qty
            existing.avg_price = new_avg
            existing.peak_price = max(existing.peak_price, price)
        # сразу персистим
        self.save()

    def on_sell(self, ticker: str, qty: float) -> None:
        """Уменьшаем позицию после SELL."""
        p = self.positions.get(ticker)
        if p is None:
            return
        p.qty = max(0.0, p.qty - qty)
        if p.qty <= 1e-9:
            # полный выход — забываем позицию
            self.positions.pop(ticker, None)
            self._half_taken.pop(ticker, None)
        self.save()

    def update_peak(self, ticker: str, price: float) -> None:
        """Обновить peak_price (вызывается каждый тик)."""
        p = self.positions.get(ticker)
        if p is None:
            return
        if price > p.peak_price:
            p.peak_price = price

    def is_buy_blocked(self, ticker: str, now_ts: Optional[float] = None) -> bool:
        """
        Заблокирован ли BUY этого тикера cooldown'ом после stop-loss?
        Вызывается ПЕРЕД принятием решения о BUY.
        """
        import time
        if ticker not in self._stop_loss_cooldown:
            return False
        last_stop_ts = self._stop_loss_cooldown[ticker]
        now = now_ts if now_ts is not None else time.time()
        return (now - last_stop_ts) < self.cooldown_seconds

    # ─────── главная проверка ───────

    def check_position(self, ticker: str, price_now: float, now_ts: Optional[float] = None) -> ExitDecision:
        """
        Каждый тик: проверяем нужно ли выходить из позиции.

        Args:
            ticker: код бумаги
            price_now: текущая цена
            now_ts: timestamp в секундах (для backtest — bar.ts, для прода — None=time.time())

        Возвращает ExitDecision. Если should_exit=True — Executor должен
        форсированно отправить SELL (full или half в зависимости от fraction).
        """
        import time as _time
        _now = now_ts if now_ts is not None else _time.time()
        p = self.positions.get(ticker)
        if p is None or p.qty <= 0 or p.avg_price <= 0:
            return ExitDecision(should_exit=False)

        # обновляем peak
        if price_now > p.peak_price:
            p.peak_price = price_now

        pnl_from_entry = (price_now - p.avg_price) / p.avg_price
        pnl_from_peak = (price_now - p.peak_price) / p.peak_price

        # 1) Stop-loss абсолютный
        if pnl_from_entry <= -self.stop_loss_pct:
            # Активируем cooldown — BUY этого тикера блокируется на N секунд
            self._stop_loss_cooldown[ticker] = _now
            return ExitDecision(
                should_exit=True,
                fraction=1.0,
                reason=ExitReason.STOP_LOSS,
                explanation=(
                    f"stop-loss: {pnl_from_entry*100:.2f}% от avg_price "
                    f"{p.avg_price:.2f} (порог -{self.stop_loss_pct*100:.1f}%). "
                    f"BUY заблокирован на {self.cooldown_seconds}s."
                ),
            )

        # 2) Take-profit на половину (если ещё не брали)
        already_half = self._half_taken.get(ticker, False)
        if not already_half and pnl_from_entry >= self.take_profit_pct:
            self._half_taken[ticker] = True
            return ExitDecision(
                should_exit=True,
                fraction=0.5,
                reason=ExitReason.TAKE_PROFIT_HALF,
                explanation=(
                    f"take-profit half: {pnl_from_entry*100:.2f}% от avg_price"
                ),
            )

        # 3) Trailing stop (только после достижения trailing_start_pct)
        if pnl_from_entry >= self.trailing_start_pct:
            if pnl_from_peak <= -self.trailing_distance_pct:
                return ExitDecision(
                    should_exit=True,
                    fraction=1.0,
                    reason=ExitReason.TRAILING_STOP,
                    explanation=(
                        f"trailing: {pnl_from_peak*100:.2f}% от peak "
                        f"{p.peak_price:.2f} (порог -{self.trailing_distance_pct*100:.1f}%)"
                    ),
                )

        # 4) Hard drawdown от peak — независимо от absolute PnL
        # Например, цена сильно выросла, потом резкий откат
        if p.peak_price > p.avg_price * 1.02:  # был хоть какой-то рост
            if pnl_from_peak <= -self.hard_drawdown_pct:
                return ExitDecision(
                    should_exit=True,
                    fraction=1.0,
                    reason=ExitReason.HARD_DRAWDOWN,
                    explanation=(
                        f"hard drawdown {pnl_from_peak*100:.2f}% от peak"
                    ),
                )

        return ExitDecision(should_exit=False)
