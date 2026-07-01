"""Единый источник параметров торговой стратегии honey_money.

Раньше эти числа были разбросаны по honey_money_provider.py, main.py и
risk_pm.py как магические литералы, и значения расходились между файлами
(например stop-loss был и 0.03, и 0.024 одновременно). Теперь все пороги,
размеры и лимиты живут в одном dataclass — это убирает рассинхрон и делает
стратегию воспроизводимой (требование ТЗ).

Любой параметр можно переопределить из окружения через ``StrategyParams.from_env``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class StrategyParams:
    """Все настраиваемые параметры decision-провайдера в одном месте."""

    # ── Капитал и размеры позиций ──
    starting_capital_rub: float = 500_000.0   # стартовый депозит по ТЗ
    min_cash_reserve: float = 10_000.0         # неснижаемый остаток кэша
    max_position_pct: float = 0.20             # макс. доля портфеля на один тикер
    order_size_pct: float = 0.15               # доля cash на одну открывающую сделку
    min_order_rub: float = 5_000.0             # анти-«москитные» сделки в 1 лот

    # ── Стоп-лосс / тейк-профит (mean-reversion профиль) ──
    default_stop_pct: float = 0.024            # -2.4% стоп
    default_take_pct: float = 0.015            # +1.5% тейк

    # ── Анти-овертрейдинг ──
    min_hold_seconds: int = 10 * 60            # минимальное удержание перед новой сделкой
    anti_avg_threshold: float = 0.001          # запрет усреднения против позиции

    # ── Volatility targeting ──
    target_vol: float = 0.02                   # целевая дневная волатильность позиции
    vol_mult_min: float = 0.5                  # нижняя граница множителя сайзинга
    vol_mult_max: float = 1.5                  # верхняя граница (cap после инцидента MOEX)

    # ── Кэш внешних данных (сек) ──
    news_fetch_interval_sec: float = 1800.0    # обновляем заголовки раз в 30 мин
    imoex_fetch_interval_sec: float = 3600.0   # IMOEX дневные свечи раз в час
    news_fetch_timeout_sec: float = 30.0       # таймаут на сбор новостей

    @property
    def hard_max_order_rub(self) -> float:
        """Жёсткий потолок одной сделки в рублях (привязан к стартовому капиталу,
        а не к раздутому шортами cash на ArenaGo)."""
        return self.starting_capital_rub * self.max_position_pct

    @classmethod
    def from_env(cls) -> "StrategyParams":
        """Собирает параметры с учётом переопределений из переменных окружения."""
        return cls(
            starting_capital_rub=_float_env("STARTING_CAPITAL_RUB", cls.starting_capital_rub),
            min_cash_reserve=_float_env("MIN_CASH_RESERVE", cls.min_cash_reserve),
            max_position_pct=_float_env("MAX_ORDER_CASH_SHARE", cls.max_position_pct),
            order_size_pct=_float_env("ORDER_SIZE_PCT", cls.order_size_pct),
            default_stop_pct=_float_env("DEFAULT_STOP_PCT", cls.default_stop_pct),
            default_take_pct=_float_env("DEFAULT_TAKE_PCT", cls.default_take_pct),
        )
