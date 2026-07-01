"""FastAPI health-check для контейнера."""

# Закрытая часть эксплуатации: /healthz нужен для локальной проверки и
# мониторинга контейнера у организаторов.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from src.arenago_client import ArenaGoClient
from src.state import StateStore


class HealthService:
    """Собирает диагностический статус контейнера."""

    def __init__(
        self,
        client: ArenaGoClient,
        state_store: StateStore,
        data_dir: Path,
        trading_enabled: bool = False,
        decision_provider_name: str = "unknown",
    ):
        """Создает health-service с runtime-метаданными процесса."""

        self.client = client
        self.state_store = state_store
        self.data_dir = data_dir
        self.trading_enabled = trading_enabled
        self.decision_provider_name = decision_provider_name
        self.started_at = datetime.now(timezone.utc)

    def check(self) -> dict[str, Any]:
        """Проверяет `/data`, checkpoint и доступность ArenaGo."""

        checks: dict[str, Any] = {
            "started_at": self.started_at.isoformat(),
            "data_dir": str(self.data_dir),
            "state_path": str(self.state_store.state_path),
            "data_dir_exists": self.data_dir.exists(),
            "trading_enabled": self.trading_enabled,
            "decision_provider": self.decision_provider_name,
        }

        status = "ok"
        try:
            state = self.state_store.load()
            checks["state"] = {
                "updated_at": state.updated_at.isoformat(),
                "safe_mode": state.safe_mode,
                "active_stops": len(state.active_stops),
                "last_error": state.last_error,
            }
        except Exception as error:
            status = "degraded"
            checks["state_error"] = str(error)

        try:
            bots = self.client.get_bots()
            checks["arenago"] = {"ok": True, "bots": [bot.name for bot in bots]}
        except Exception as error:
            status = "degraded"
            checks["arenago"] = {"ok": False, "error": str(error)}

        checks["status"] = status
        checks["checked_at"] = datetime.now(timezone.utc).isoformat()
        return checks


def create_app(service: HealthService) -> FastAPI:
    """Создает FastAPI-приложение с endpoint `/healthz`."""

    app = FastAPI(title="MOEX AI Bot Health")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Возвращает 200 для healthy runtime и 503 для degraded runtime."""

        result = service.check()
        if result["status"] != "ok":
            raise HTTPException(status_code=503, detail=result)
        return result

    return app
