"""
Macro context loader для MacroContextMLProvider.

Читает config/macro_priors.yaml и предоставляет typed-структуру:
  - MacroState        : текущий снимок макро-факторов (boolean флаги)
  - MacroEvent        : событие в календаре (PCE, отсечка дивов, …)
  - TickerPriors      : per-ticker реакции на каждый макро-фактор
  - OverlayConfig     : параметры применения overlay
  - MacroContext      : композит всех четырёх + методы расчёта logit-shift

ВАЖНО: загрузка делается один раз при старте бота. Если файл правится
в рантайме — нужно дёрнуть `MacroContext.reload()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ────────────────────────── Data classes ──────────────────────────


@dataclass(frozen=True)
class MacroState:
    """Текущий снимок макро-факторов как boolean-флаги."""
    flags: Dict[str, bool]
    raw: Dict[str, Any] = field(default_factory=dict)  # нечисловые сырые значения

    def is_active(self, flag: str) -> bool:
        return bool(self.flags.get(flag, False))


@dataclass
class MacroEvent:
    """Событие в календаре. Время хранится в UTC."""
    id: str
    name: str
    when_utc: datetime
    binary: bool
    affects_tickers: List[str]
    pre_event_hold_hours: float = 0.0
    post_event_hold_minutes: float = 0.0
    pre_event_logit_shift: float = 0.0
    prior_logit_shift: float = 0.0
    decay_days: float = 0.0
    outcome: Optional[str] = None
    outcome_shifts: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    # ─────── temporal helpers ───────

    def is_holding_window(self, now: datetime) -> bool:
        """T-pre_event_hold_hours .. T+post_event_hold_minutes — HOLD."""
        if not self.binary:
            return False
        start = self.when_utc - timedelta(hours=self.pre_event_hold_hours)
        end = self.when_utc + timedelta(minutes=self.post_event_hold_minutes)
        return start <= now <= end

    def effective_logit_shift(self, now: datetime) -> float:
        """
        Возвращает текущий логит-сдвиг для этого события:
          - до релиза с time-decay (для не-бинарных) или prior_shift (для бинарных)
          - после релиза — outcome-shift из outcome_shifts[outcome] если задан
        """
        # Бинарные события
        if self.binary:
            if now < self.when_utc:
                # До релиза — закладываем ожидание аналитика
                return self.prior_logit_shift
            if self.outcome and self.outcome in self.outcome_shifts:
                return self.outcome_shifts[self.outcome]
            # после релиза, но outcome ещё не выставлен — гасим тилт
            return 0.0

        # Не-бинарные (дивотсечка, дедлайн санкций)
        if now >= self.when_utc:
            # событие прошло — overlay сам по себе не должен жить вечно
            return 0.0
        days_to = (self.when_utc - now).total_seconds() / 86400.0
        if self.decay_days <= 0:
            return self.pre_event_logit_shift
        # линейный ramp: 0 за decay_days дней, full shift в день события
        ramp = max(0.0, min(1.0, 1.0 - days_to / self.decay_days))
        return self.pre_event_logit_shift * ramp


@dataclass(frozen=True)
class TickerPriors:
    """Per-ticker реакция на флаги и привязка к событию."""
    base_bias: float
    factors: Dict[str, float]
    pce_event_id: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class OverlayConfig:
    max_logit_shift: float = 0.8
    base_provider_weight: float = 1.0
    enable_event_gating: bool = True
    log_decompose: bool = True
    default_unknown_ticker_passthrough: bool = True


# ────────────────────────── MacroContext ──────────────────────────


@dataclass
class LogitDecomposition:
    """Прозрачное разложение применённого логит-сдвига."""
    base_bias: float = 0.0
    flag_contributions: Dict[str, float] = field(default_factory=dict)
    event_contributions: Dict[str, float] = field(default_factory=dict)
    total_before_clip: float = 0.0
    total: float = 0.0  # после клиппинга
    holding_event_id: Optional[str] = None  # если форсируем HOLD

    def to_rationale(self) -> str:
        parts = [f"base={self.base_bias:+.3f}"]
        for flag, val in self.flag_contributions.items():
            if abs(val) > 1e-6:
                parts.append(f"{flag}={val:+.3f}")
        for ev, val in self.event_contributions.items():
            if abs(val) > 1e-6:
                parts.append(f"ev:{ev}={val:+.3f}")
        if self.holding_event_id:
            parts.append(f"HOLD-WINDOW:{self.holding_event_id}")
        parts.append(f"Σ={self.total:+.3f}")
        return " ".join(parts)


class MacroContext:
    """
    Контекст макро-приоров. Подгружает YAML, считает логит-сдвиг
    по тикеру и времени, экспортирует объяснение.
    """

    def __init__(self, priors_path: str = "config/macro_priors.yaml") -> None:
        self.priors_path = Path(priors_path)
        self._loaded = False
        self.state: MacroState = MacroState(flags={})
        self.events: Dict[str, MacroEvent] = {}
        self.tickers: Dict[str, TickerPriors] = {}
        self.overlay_cfg: OverlayConfig = OverlayConfig()
        self.asof: Optional[datetime] = None
        self.reload()

    # ─────── public API ───────

    def reload(self) -> None:
        if not self.priors_path.exists():
            # Файл может отсутствовать в тестах — оставляем пустой контекст
            self._loaded = False
            return
        with open(self.priors_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._parse(data)
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def is_holding_window_for(self, ticker: str, now: datetime) -> Optional[str]:
        """Если ticker сейчас в HOLD-окне какого-то бинарного события — вернёт его id."""
        if not self.overlay_cfg.enable_event_gating:
            return None
        priors = self.tickers.get(ticker)
        if priors is None:
            return None
        for event in self.events.values():
            if ticker not in event.affects_tickers:
                continue
            if event.is_holding_window(now):
                return event.id
        return None

    def compute_logit_shift(
        self,
        ticker: str,
        now: datetime,
    ) -> LogitDecomposition:
        """
        Считает совокупный logit-shift для тикера, разложенный по факторам.

        Логика:
          base_bias + Σ flag_contrib + Σ event_contrib
        После расчёта клиппится в [-max_logit_shift, +max_logit_shift].
        """
        decomp = LogitDecomposition()

        priors = self.tickers.get(ticker)
        if priors is None:
            # неизвестный тикер
            return decomp

        decomp.base_bias = priors.base_bias

        # вклад флагов macro_state
        for flag_name, coef in priors.factors.items():
            if self.state.is_active(flag_name):
                decomp.flag_contributions[flag_name] = coef

        # вклад событий календаря (где этот тикер указан в affects_tickers)
        for event in self.events.values():
            if ticker not in event.affects_tickers:
                continue
            shift = event.effective_logit_shift(now)
            if abs(shift) > 1e-9:
                decomp.event_contributions[event.id] = shift

        raw_sum = (
            decomp.base_bias
            + sum(decomp.flag_contributions.values())
            + sum(decomp.event_contributions.values())
        )
        decomp.total_before_clip = raw_sum

        cap = self.overlay_cfg.max_logit_shift
        decomp.total = max(-cap, min(cap, raw_sum))

        # HOLD-окно — на этой стадии не зануляем сдвиг (это делает overlay
        # выше уровнем), но прокидываем id события в decomposition
        hold_id = self.is_holding_window_for(ticker, now)
        if hold_id:
            decomp.holding_event_id = hold_id

        return decomp

    # ─────── parsing ───────

    def _parse(self, data: Dict[str, Any]) -> None:
        # asof
        if "asof" in data:
            try:
                self.asof = datetime.fromisoformat(data["asof"]).astimezone(timezone.utc)
            except Exception:
                self.asof = None

        # macro_state — разделяем bool / non-bool
        raw_state = data.get("macro_state", {}) or {}
        flags: Dict[str, bool] = {}
        raw: Dict[str, Any] = {}
        for k, v in raw_state.items():
            if isinstance(v, bool):
                flags[k] = v
            else:
                raw[k] = v
        self.state = MacroState(flags=flags, raw=raw)

        # events
        events: Dict[str, MacroEvent] = {}
        for ev in data.get("events", []) or []:
            try:
                when = datetime.fromisoformat(ev["when"]).astimezone(timezone.utc)
            except Exception:
                # пропускаем некорректные записи, не падаем
                continue
            outcome_shifts: Dict[str, float] = {}
            for key in (
                "outcome_logit_shift_hot",
                "outcome_logit_shift_soft",
                "outcome_logit_shift_as_expected",
                "outcome_logit_shift_dovish",
                "outcome_logit_shift_hawkish",
                "outcome_logit_shift_neutral",
            ):
                if key in ev:
                    bucket = key.replace("outcome_logit_shift_", "")
                    outcome_shifts[bucket] = float(ev[key])

            events[ev["id"]] = MacroEvent(
                id=ev["id"],
                name=ev.get("name", ev["id"]),
                when_utc=when,
                binary=bool(ev.get("binary", False)),
                affects_tickers=list(ev.get("affects_tickers", []) or []),
                pre_event_hold_hours=float(ev.get("pre_event_hold_hours", 0.0)),
                post_event_hold_minutes=float(ev.get("post_event_hold_minutes", 0.0)),
                pre_event_logit_shift=float(ev.get("pre_event_logit_shift", 0.0)),
                prior_logit_shift=float(ev.get("prior_logit_shift", 0.0)),
                decay_days=float(ev.get("decay_days", 0.0)),
                outcome=ev.get("outcome"),
                outcome_shifts=outcome_shifts,
                notes=ev.get("notes", ""),
            )
        self.events = events

        # tickers
        tickers: Dict[str, TickerPriors] = {}
        for tk, tdata in (data.get("tickers", {}) or {}).items():
            tickers[tk] = TickerPriors(
                base_bias=float(tdata.get("base_bias", 0.0)),
                factors={k: float(v) for k, v in (tdata.get("factors", {}) or {}).items()},
                pce_event_id=tdata.get("pce_event_id"),
                notes=tdata.get("notes", ""),
            )
        self.tickers = tickers

        # overlay config
        ovc = data.get("overlay_config", {}) or {}
        self.overlay_cfg = OverlayConfig(
            max_logit_shift=float(ovc.get("max_logit_shift", 0.8)),
            base_provider_weight=float(ovc.get("base_provider_weight", 1.0)),
            enable_event_gating=bool(ovc.get("enable_event_gating", True)),
            log_decompose=bool(ovc.get("log_decompose", True)),
            default_unknown_ticker_passthrough=bool(
                ovc.get("default_unknown_ticker_passthrough", True)
            ),
        )


# ─────────── Logit helpers (общие, используются в провайдере) ────


def prob_to_logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def logit_to_prob(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))
