"""
CrisisGate — защита от каскадных потерь при обвале рынка.

Логика:
  - Считаем return IMOEX за последние 60 минут (можно настроить)
  - Если return ≤ -2% → state = CRISIS
  - В CRISIS:
      * force_sell_all_longs (executor должен закрыть всё)
      * блокируем новые BUY на cooldown_minutes (по умолчанию 60 мин)
  - После cooldown — возвращаемся к normal

Используется в:
  - main цикле перед SignalEngine: если CRISIS → action = SELL_ALL для всех
  - backtest_hybrid.py — то же самое

Состояние сохраняется в /data/crisis_state.json для recovery.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class CrisisState(str, Enum):
    NORMAL = "normal"
    CRISIS = "crisis"          # активная фаза — закрываем всё, BUY заблокирован
    COOLDOWN = "cooldown"      # после кризиса, ждём остывания


@dataclass
class CrisisLog:
    """Запись о случае срабатывания."""
    ts: str
    imoex_return_1h: float
    triggered_state: str


class CrisisGate:
    """
    Сторож-наблюдатель за IMOEX. По умолчанию срабатывает при -2% за 1ч.
    После активации блокирует новые long-позиции на 60 минут.
    """

    def __init__(
        self,
        trigger_pct: float = 0.02,             # -2% за окно
        window_minutes: int = 60,              # окно отсчёта
        cooldown_minutes: int = 60,            # сколько ждать после активации
        state_path: str = "/data/crisis_state.json",
    ) -> None:
        self.trigger_pct = trigger_pct
        self.window_minutes = window_minutes
        self.cooldown_minutes = cooldown_minutes
        self.state_path = Path(state_path)

        self.state = CrisisState.NORMAL
        self.last_trigger_ts: Optional[datetime] = None
        self.history: List[CrisisLog] = []
        self._load()

    # ─────── persistence ───────

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state = CrisisState(data.get("state", "normal"))
            tts = data.get("last_trigger_ts")
            if tts:
                self.last_trigger_ts = datetime.fromisoformat(tts)
            self.history = [CrisisLog(**h) for h in data.get("history", [])]
        except Exception as e:
            logger.warning("CrisisGate: state load failed: %s", e)

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "state": self.state.value,
                "last_trigger_ts": self.last_trigger_ts.isoformat() if self.last_trigger_ts else None,
                "history": [asdict(h) for h in self.history[-50:]],  # последние 50
            }
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("CrisisGate: state save failed: %s", e)

    # ─────── API ───────

    def update(
        self,
        imoex_closes_1h: List[float],
        now: Optional[datetime] = None,
    ) -> CrisisState:
        """
        Главный апдейт. На вход — список закрытий IMOEX за последние N часов
        (нам нужно минимум 2 для отсчёта).
        """
        now = now or datetime.now(timezone.utc)

        # 1) Возможно ли выйти из cooldown'а?
        if self.state == CrisisState.CRISIS or self.state == CrisisState.COOLDOWN:
            if self.last_trigger_ts and (now - self.last_trigger_ts) >= timedelta(
                minutes=self.cooldown_minutes
            ):
                logger.info("CrisisGate: cooldown over → NORMAL")
                self.state = CrisisState.NORMAL
                self.save()

        # 2) Считаем return за окно
        if not imoex_closes_1h or len(imoex_closes_1h) < 2:
            return self.state

        # Считаем return от первого до последнего close в окне
        first = imoex_closes_1h[0]
        last = imoex_closes_1h[-1]
        if first <= 0:
            return self.state
        ret = (last - first) / first

        # 3) Триггер?
        if ret <= -self.trigger_pct and self.state == CrisisState.NORMAL:
            logger.warning("CrisisGate TRIGGERED: IMOEX %.2f%% за окно", ret * 100)
            self.state = CrisisState.CRISIS
            self.last_trigger_ts = now
            self.history.append(CrisisLog(
                ts=now.isoformat(),
                imoex_return_1h=ret,
                triggered_state="CRISIS",
            ))
            self.save()

        return self.state

    def should_force_sell_longs(self) -> bool:
        """Executor должен закрыть все long позиции?"""
        return self.state == CrisisState.CRISIS

    def is_buy_blocked(self) -> bool:
        """Заблокированы ли новые BUY?"""
        return self.state in (CrisisState.CRISIS, CrisisState.COOLDOWN)

    def reset(self) -> None:
        """Аварийный сброс (ручной). НЕ ИСПОЛЬЗОВАТЬ в автономном режиме!"""
        self.state = CrisisState.NORMAL
        self.last_trigger_ts = None
        self.save()
