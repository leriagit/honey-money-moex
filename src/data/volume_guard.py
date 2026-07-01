"""
VolumeGuard — следит за суммарным торговым оборотом за этап 2.

КРИТИЧНО (правила ТЗ, штраф −70 баллов из 70):
  > Суммарный торговый оборот агента по итогам автономных тестов (этап 2)
  > должен составлять более 10 000 000 руб.

Если бот вёл себя слишком осторожно (96% HOLD) — мы можем закончить этап 2
с PnL +5%, но оборотом всего 3М₽, и тогда автоматически потеряем 70 баллов.

Решение: VolumeGuard следит за текущим оборотом, и если к концу периода
он недостаточен — повышает urgency сигналов (понижает порог BUY/SELL,
форсирует rotation между позициями).

Логика:
  - Стартовая дата этапа 2 (autonomous): 28 мая 2026, 07:00 МСК
  - Окончание: 10 июня 2026, 15:00 МСК
  - Целевой минимум оборота: 10_000_000 ₽ + buffer 15% = 11_500_000 ₽
  - Каждый час вычисляем "required pace" = (target - current) / hours_left
  - Если current_pace < required_pace * 0.7 — поднимаем urgency

Использование (в SignalEngine):
  guard = VolumeGuard()
  ...
  if guard.urgency_level() > 0:
      # понизить пороги BUY_SMALL / SELL_HALF, чтобы бот активнее торговал
      threshold_shift -= guard.urgency_level() * 0.05
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────── Конфиг этапа 2 ───────────────
# Можно переопределить через env, по умолчанию — даты из ТЗ.

STAGE2_START_UTC = datetime(2026, 5, 28, 4, 0, tzinfo=timezone.utc)  # 07:00 МСК
STAGE2_END_UTC = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)   # 15:00 МСК
TARGET_TURNOVER_RUB = 10_000_000.0
SAFETY_BUFFER = 1.15  # хотим набрать 115% от минимума на всякий случай


@dataclass
class TurnoverState:
    """Состояние, сохраняется в /data/turnover.json для recovery."""
    total_buy_rub: float = 0.0
    total_sell_rub: float = 0.0
    trades_count: int = 0
    last_trade_ts: Optional[str] = None  # ISO-строка
    trade_log: List[dict] = field(default_factory=list)  # последние N трейдов

    @property
    def total_turnover(self) -> float:
        """Оборот = buys + sells (двусторонний)."""
        return self.total_buy_rub + self.total_sell_rub


class VolumeGuard:
    """
    Следит за оборотом, выдаёт urgency-сигнал для SignalEngine.

    State persistence: пишем в /data/turnover.json — переживает деплои.
    """

    def __init__(
        self,
        state_path: str = "/data/turnover.json",
        stage_start: Optional[datetime] = None,
        stage_end: Optional[datetime] = None,
        target_rub: float = TARGET_TURNOVER_RUB,
        safety_buffer: float = SAFETY_BUFFER,
    ) -> None:
        self.state_path = Path(state_path)
        self.stage_start = stage_start or STAGE2_START_UTC
        self.stage_end = stage_end or STAGE2_END_UTC
        self.target_rub = target_rub
        self.target_with_buffer = target_rub * safety_buffer

        self.state = self._load_state()

    # ─────────── persistence ───────────

    def _load_state(self) -> TurnoverState:
        if not self.state_path.exists():
            return TurnoverState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return TurnoverState(**data)
        except Exception as e:
            logger.warning("VolumeGuard: не смогли загрузить state из %s: %s — начинаем с нуля",
                           self.state_path, e)
            return TurnoverState()

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(asdict(self.state), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("VolumeGuard: не смогли сохранить state в %s: %s",
                         self.state_path, e)

    # ─────────── публичный API ───────────

    def register_trade(
        self,
        side: str,
        ticker: str,
        quantity: int,
        price: float,
        ts: Optional[datetime] = None,
    ) -> None:
        """
        Зарегистрировать совершённую сделку. Вызывается ИЗ Executor'а после
        успешного fill'а от ArenaGo.
        """
        ts = ts or datetime.now(timezone.utc)
        value = abs(quantity) * price

        if side.upper() in ("B", "BUY"):
            self.state.total_buy_rub += value
        elif side.upper() in ("S", "SELL"):
            self.state.total_sell_rub += value
        else:
            logger.warning("VolumeGuard: unknown side '%s' for %s", side, ticker)
            return

        self.state.trades_count += 1
        self.state.last_trade_ts = ts.isoformat()

        # Лог последних 100 сделок — для аудита
        self.state.trade_log.append({
            "ts": ts.isoformat(),
            "side": side.upper(),
            "ticker": ticker,
            "qty": int(quantity),
            "price": float(price),
            "value": float(value),
        })
        if len(self.state.trade_log) > 100:
            self.state.trade_log = self.state.trade_log[-100:]

        self.save()
        logger.info("VolumeGuard: registered %s %s x%d @ %.2f = %.0f ₽ (total turnover %.0f)",
                    side, ticker, quantity, price, value, self.state.total_turnover)

    def current_turnover(self) -> float:
        return self.state.total_turnover

    def progress(self) -> float:
        """Доля выполнения цели (с buffer'ом): 0.0 → 1.0+."""
        return self.state.total_turnover / self.target_with_buffer

    def required_pace_rub_per_hour(self, now: Optional[datetime] = None) -> float:
        """Сколько ещё нужно нагенерить оборота в час, чтобы успеть к концу периода."""
        now = now or datetime.now(timezone.utc)
        if now >= self.stage_end:
            return 0.0
        if now < self.stage_start:
            # этап ещё не начался
            hours_total = (self.stage_end - self.stage_start).total_seconds() / 3600.0
            return self.target_with_buffer / hours_total
        remaining = self.target_with_buffer - self.state.total_turnover
        if remaining <= 0:
            return 0.0
        hours_left = (self.stage_end - now).total_seconds() / 3600.0
        if hours_left <= 0:
            return float("inf")  # критично
        return remaining / hours_left

    def actual_pace_rub_per_hour(self, now: Optional[datetime] = None) -> float:
        """Текущий темп оборота — в час, с момента начала этапа."""
        now = now or datetime.now(timezone.utc)
        if now < self.stage_start:
            return 0.0
        hours_passed = (now - self.stage_start).total_seconds() / 3600.0
        if hours_passed < 0.5:
            return 0.0  # слишком рано судить
        return self.state.total_turnover / hours_passed

    def urgency_level(self, now: Optional[datetime] = None) -> float:
        """
        Возвращает 0.0 .. 1.0 — насколько срочно нужно увеличивать активность.

          0.0 — мы в графике (или цель достигнута)
          0.3 — пока ок, но темп немного отстаёт
          0.7 — нужно увеличить активность
          1.0 — критично, осталось мало времени

        Используется в SignalEngine для динамической подстройки порогов.
        """
        now = now or datetime.now(timezone.utc)

        # Цель достигнута — нет urgency
        if self.state.total_turnover >= self.target_with_buffer:
            return 0.0
        # Вне этапа 2 — нет смысла
        if now < self.stage_start or now >= self.stage_end:
            return 0.0

        required = self.required_pace_rub_per_hour(now)
        actual = self.actual_pace_rub_per_hour(now)

        if actual >= required:
            return 0.0
        if actual <= 0:
            # вообще ничего не наторговали — считаем по времени до конца
            elapsed_pct = (now - self.stage_start) / (self.stage_end - self.stage_start)
            return min(1.0, elapsed_pct * 1.5)
        ratio = actual / required
        # 0.7 → urgency 0.3; 0.5 → urgency 0.5; 0.2 → urgency 0.8
        return min(1.0, max(0.0, 1.0 - ratio))

    def status_report(self, now: Optional[datetime] = None) -> dict:
        """Сводка для health-endpoint и audit-log."""
        now = now or datetime.now(timezone.utc)
        return {
            "turnover_rub": round(self.state.total_turnover, 2),
            "target_rub": self.target_rub,
            "target_with_buffer_rub": self.target_with_buffer,
            "progress_pct": round(self.progress() * 100, 2),
            "trades_count": self.state.trades_count,
            "last_trade_ts": self.state.last_trade_ts,
            "now": now.isoformat(),
            "stage2_start": self.stage_start.isoformat(),
            "stage2_end": self.stage_end.isoformat(),
            "hours_remaining": max(0.0, (self.stage_end - now).total_seconds() / 3600.0),
            "required_pace_rub_per_hour": round(self.required_pace_rub_per_hour(now), 0),
            "actual_pace_rub_per_hour": round(self.actual_pace_rub_per_hour(now), 0),
            "urgency_level": round(self.urgency_level(now), 3),
            "penalty_risk": self.state.total_turnover < self.target_rub,
        }
