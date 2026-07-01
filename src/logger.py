"""JSON-логи и append-only audit.log."""

# Закрытая часть audit: сюда пишется доказуемая история решений, заявок,
# dry-run и ошибок для последующей экспертной проверки.

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Форматтер одной JSONL-строки для stdout и файла логов."""

    def format(self, record: logging.LogRecord) -> str:
        """Преобразует `LogRecord` в JSON-строку."""

        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(data_dir: Path) -> logging.Logger:
    """Настраивает логирование в stdout и `/data/logs/app.jsonl`."""

    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("moex_ai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = JsonFormatter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(logs_dir / "app.jsonl")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def write_audit_event(data_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    """Добавляет событие в append-only `/data/audit.log`."""

    data_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with (data_dir / "audit.log").open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
