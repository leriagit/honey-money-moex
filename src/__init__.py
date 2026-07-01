"""Инфраструктурный пакет MOEX AI bot.

При импорте автоматически подгружает .env из корня репо в os.environ,
если файл существует и переменные ещё не заданы. Это позволяет работать
без явного python-dotenv или ручной загрузки.

Безопасно: переменные из os.environ имеют приоритет над .env.
"""
import os
from pathlib import Path


def _auto_load_env() -> None:
    """Подгружает .env из корня репо если он есть. Тихо, без ошибок."""
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Системные env имеют приоритет
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Тихо: проблемы с .env не должны падать импорт
        pass


_auto_load_env()
