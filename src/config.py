"""Env-конфигурация контейнера."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Фиксированная вселенная тикеров по условиям соревнования.
TICKERS: tuple[str, ...] = (
    "LKOH",
    "SBER",
    "ROSN",
    "GAZP",
    "VTBR",
    "YDEX",
    "PLZL",
    "T",
    "NVTK",
    "X5",
    "GMKN",
    "MGNT",
    "ALRS",
    "AFLT",
    "CHMF",
    "NLMK",
    "MOEX",
    "SNGSP",
    "MTSS",
    "PIKK",
)


@dataclass(frozen=True)
class Settings:
    """Снимок runtime-настроек, прочитанный из переменных окружения."""

    arenago_api_key: str
    arenago_base_url: str
    bot_name: str
    data_dir: Path
    decision_interval_seconds: int
    stop_watch_interval_seconds: float
    health_host: str
    health_port: int
    request_timeout_seconds: float
    request_max_retries: int
    min_cash_reserve: float
    max_order_cash_share: float
    max_daily_trades: int
    allow_shorts: bool
    trading_enabled: bool


def _bool_env(name: str, default: bool) -> bool:
    """Парсит булеву переменную окружения с безопасным значением по умолчанию."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    """Парсит целочисленную переменную окружения и явно сообщает об ошибке формата."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error


def _float_env(name: str, default: float) -> float:
    """Парсит числовую переменную окружения и явно сообщает об ошибке формата."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error


def load_settings(require_api_key: bool = True) -> Settings:
    """Собирает настройки контейнера в единый объект `Settings`."""

    api_key = os.getenv("SANDBOX_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError("SANDBOX_API_KEY is required for ArenaGo API access")

    # ВАЖНО: дефолты подобраны так, чтобы бот работал даже без CI/CD Variables
    # на сервере organizers. Если organizers выставят свои переменные —
    # они перекроют эти дефолты автоматически.
    #
    # BOT_NAME: наш портфель на ArenaGo называется "honey money" (с пробелом)
    # TRADING_ENABLED: на серверном деплое всегда true (это и есть цель бота)
    return Settings(
        arenago_api_key=api_key,
        arenago_base_url=os.getenv("ARENAGO_BASE_URL", "https://arenago.ru"),
        bot_name=os.getenv("BOT_NAME", "honey money"),
        data_dir=Path(os.getenv("DATA_DIR", "/data")),
        decision_interval_seconds=_int_env("DECISION_INTERVAL_SECONDS", 600),
        stop_watch_interval_seconds=_float_env("STOP_WATCH_INTERVAL_SECONDS", 3),
        health_host=os.getenv("HEALTH_HOST", "0.0.0.0"),
        health_port=_int_env("HEALTH_PORT", 8080),
        request_timeout_seconds=_float_env("REQUEST_TIMEOUT_SECONDS", 10),
        request_max_retries=_int_env("REQUEST_MAX_RETRIES", 3),
        min_cash_reserve=_float_env("MIN_CASH_RESERVE", 10000),
        max_order_cash_share=_float_env("MAX_ORDER_CASH_SHARE", 0.20),
        max_daily_trades=_int_env("MAX_DAILY_TRADES", 1000),
        allow_shorts=_bool_env("ALLOW_SHORTS", True),
        trading_enabled=_bool_env("TRADING_ENABLED", True),
    )
