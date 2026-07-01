"""Checkpoint бота в `/data/state.json`."""

# Закрытая часть recovery: состояние процесса хранится в /data и сверяется
# с ArenaGo как источником истины после старта и каждого цикла.

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from src.schemas import BotState, Position, StopSpec, Trade


class StateStore:
    """Потокобезопасное файловое хранилище checkpoint-состояния."""

    def __init__(self, data_dir: Path):
        """Создает store для файла `/data/state.json` или его локального аналога."""

        self.data_dir = data_dir
        self.state_path = data_dir / "state.json"
        self._lock = RLock()

    def load(self) -> BotState:
        """Загружает checkpoint или возвращает пустое состояние при первом старте."""

        with self._lock:
            return self._load_unlocked()

    def save(self, state: BotState) -> None:
        """Атомарно сохраняет checkpoint через временный файл и `os.replace`."""

        with self._lock:
            self._save_unlocked(state)

    def reconcile_from_snapshots(
        self,
        cash_balance: float,
        positions: list[Position],
        trades_today: list[Trade],
    ) -> BotState:
        """Обновляет checkpoint по фактическим данным ArenaGo."""

        with self._lock:
            state = self._load_unlocked()
            state.cash_balance = cash_balance
            state.positions = {position.secid: int(position.position) for position in positions}
            state.average_prices = {
                position.secid: float(position.average_price)
                for position in positions
                if position.average_price is not None
            }
            state.daily_trade_count = len(trades_today)
            state.turnover_today = sum(trade.value for trade in trades_today)
            state.last_reconcile_at = datetime.now(timezone.utc)
            state.active_stops = {
                stop_id: stop
                for stop_id, stop in state.active_stops.items()
                if state.positions.get(stop.ticker, 0) != 0
            }
            self._save_unlocked(state)
            return state

    def add_stop(self, stop: StopSpec) -> BotState:
        """Добавляет виртуальный стоп или тейк в checkpoint."""

        with self._lock:
            state = self._load_unlocked()
            state.active_stops[stop.stop_id] = stop
            self._save_unlocked(state)
            return state

    def remove_stop(self, stop_id: str) -> BotState:
        """Удаляет виртуальный стоп или тейк из checkpoint."""

        with self._lock:
            state = self._load_unlocked()
            state.active_stops.pop(stop_id, None)
            self._save_unlocked(state)
            return state

    def record_error(self, message: str, safe_mode: bool = False) -> BotState:
        """Фиксирует последнюю runtime-ошибку в checkpoint."""

        with self._lock:
            state = self._load_unlocked()
            state.last_error = message
            state.safe_mode = safe_mode
            self._save_unlocked(state)
            return state

    def _load_unlocked(self) -> BotState:
        """Читает checkpoint без захвата lock; вызывается только под `self._lock`."""

        if not self.state_path.exists():
            return BotState()
        try:
            with self.state_path.open("r", encoding="utf-8") as state_file:
                return BotState.model_validate(json.load(state_file))
        except Exception as error:
            corrupt_path = self.state_path.with_suffix(f".corrupt.{int(datetime.now(timezone.utc).timestamp())}.json")
            os.replace(self.state_path, corrupt_path)
            return BotState(last_error=f"state_recovered_from_corrupt_file: {error}")

    def _save_unlocked(self, state: BotState) -> None:
        """Записывает checkpoint без захвата lock; вызывается только под `self._lock`."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        state.updated_at = datetime.now(timezone.utc)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as state_file:
            state_file.write(state.model_dump_json(indent=2))
        os.replace(tmp_path, self.state_path)
