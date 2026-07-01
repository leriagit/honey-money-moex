"""Общая настройка тестов.

Деплой идёт на Python 3.12 (см. Dockerfile), где datetime.UTC доступен. Чтобы
тесты гонялись и на более старом локальном Python 3.10, мягко подставляем UTC.
Это влияет ТОЛЬКО на тестовое окружение и не трогает production-код.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

if not hasattr(_dt, "UTC"):  # pragma: no cover — только для Python < 3.11
    _dt.UTC = _dt.timezone.utc

# Корень проекта в sys.path, чтобы `import src...` работал из любого каталога.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
