"""Общие Pydantic-схемы для API, состояния и исполнения."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class OrderSide(str, Enum):
    BUY = "B"
    SELL = "S"


class DecisionAction(str, Enum):
    HOLD = "hold"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    REDUCE = "reduce"
    INCREASE = "increase"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class BotInfo(FlexibleModel):
    name: str
    cash_balance: float = 0.0


class Position(FlexibleModel):
    secid: str
    position: int = 0
    average_price: float | None = None
    bot: str | None = None


class Trade(FlexibleModel):
    tradedate: str | None = None
    tradetime: str | None = None
    direction: str | None = None
    secid: str
    quantity: int
    price: float
    bot: str | None = None

    @property
    def value(self) -> float:
        """Возвращает денежный оборот сделки в абсолютном значении."""

        return abs(self.quantity * self.price)


class OrderRequest(BaseModel):
    direction: OrderSide
    secid: str
    quantity: int = Field(ge=1)
    bot: str


class OrderResponse(FlexibleModel):
    success: bool = False
    message: str | None = None
    order_value: float | None = None
    price: float | None = None
    quantity: int | None = None
    remaining_cash: float | None = None


class AgentOrder(FlexibleModel):
    """Сырой ордер от LLM/fallback-агентов (PortfolioManager) ДО риск-сайзинга.

    Раньше эти ордера ходили между оркестратором и OrderMapper как нетипизированные
    ``Dict[str, Any]`` с россыпью ``.get(...)``. Теперь это типизированная модель:
    числовые поля — Optional, чтобы OrderMapper мог подставить дефолты из
    StrategyParams (поведение и значения по умолчанию сохранены).
    """

    ticker: str
    action: str = "HOLD"
    size_pct: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    rationale: str = ""

    @classmethod
    def from_raw(cls, data: Any) -> "AgentOrder | None":
        """Толерантно парсит dict (или готовый AgentOrder) в AgentOrder.

        Возвращает None, если объект непригоден (нет тикера) — такие ордера
        отбрасываются, как и раньше.
        """
        if isinstance(data, AgentOrder):
            return data
        if not isinstance(data, dict) or not data.get("ticker"):
            return None
        try:
            return cls.model_validate(data)
        except Exception:
            return None

    @property
    def action_upper(self) -> str:
        return (self.action or "HOLD").upper()


class DecisionOrder(BaseModel):
    ticker: str
    action: DecisionAction
    size_rub: float | None = Field(default=None, gt=0)
    size_lots: int | None = Field(default=None, ge=1)
    stop_price: float | None = Field(default=None, gt=0)
    take_price: float | None = Field(default=None, gt=0)
    priority: int = 100
    reason_summary: str = ""


class DecisionPlan(BaseModel):
    orders: list[DecisionOrder] = Field(default_factory=list)
    no_action_tickers: list[str] = Field(default_factory=list)
    cycle_summary: str = "No trading action requested"
    next_cycle_in_minutes: int = 15
    mode: str = "stub"
    raw: dict[str, Any] = Field(default_factory=dict)


class StopSpec(BaseModel):
    stop_id: str
    ticker: str
    side: PositionSide
    quantity: int = Field(ge=1)
    entry_price: float = Field(gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    take_price: float | None = Field(default=None, gt=0)
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""


# ─────────────────────────────────────────────────────────────────
# Honey-money-bot extension schemas (added для интеграции ML/LLM stack)
# Эти классы используют src/data/ml_signal.py, indicators.py, news/.
# Не используются базовым контуром (executor/risk/stop_watcher) — только
# нашим decision-provider.
# ─────────────────────────────────────────────────────────────────


class Candle(BaseModel):
    """OHLCV-свеча. Используется ML-провайдером и indicators."""
    model_config = ConfigDict(frozen=True)
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorBundle(BaseModel):
    """Технические индикаторы по одному тикеру (RSI/MACD/Volume/SMA)."""
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    macd_cross_up: bool = False
    macd_cross_down: bool = False
    volume_last: float | None = None
    volume_avg_20: float | None = None
    volume_ratio: float | None = None
    sma_20: float | None = None
    price: float | None = None


class MLSignal(BaseModel):
    """Выход ML-провайдера: вероятность роста за горизонт."""
    prob_up: float = Field(ge=0.0, le=1.0)
    expected_return: float | None = None
    expected_vol: float | None = None
    horizon: str = "1h"
    model_version: str = "stub-v1"


class NewsItem(BaseModel):
    """Одна новость из источника (используется парсерами)."""
    source_id: str
    title: str
    summary: str | None = None
    url: str | None = None
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    sentiment: float = Field(ge=-1.0, le=1.0, default=0.0)
    ru_relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    trump_mention: bool = False
    raw_score: float | None = None
    regulator_event: bool = False


class NewsSignal(BaseModel):
    """Агрегированный новостной сигнал по тикеру."""
    ticker: str
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    n_items: int = 0
    n_ru_relevant: int = 0
    n_trump: int = 0
    ru_defense_flag: bool = False
    regulator_event: bool = False
    regulator_event_titles: list[str] = Field(default_factory=list)
    top_items: list[NewsItem] = Field(default_factory=list)


class BotState(BaseModel):
    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cash_balance: float = 0.0
    positions: dict[str, int] = Field(default_factory=dict)
    average_prices: dict[str, float] = Field(default_factory=dict)
    active_stops: dict[str, StopSpec] = Field(default_factory=dict)
    turnover_today: float = 0.0
    daily_trade_count: int = 0
    safe_mode: bool = False
    last_reconcile_at: datetime | None = None
    last_error: str | None = None


class ValidationResult(BaseModel):
    allowed: bool
    reason: str = "ok"


class ExecutionStatus(str, Enum):
    EXECUTED = "executed"
    DRY_RUN = "dry_run"
    REJECTED = "rejected"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionResult(BaseModel):
    order: DecisionOrder
    status: ExecutionStatus
    reason: str
    request: OrderRequest | None = None
    response: OrderResponse | None = None
