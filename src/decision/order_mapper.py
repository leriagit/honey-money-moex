"""Маппинг сырых LLM-ордеров в типизированные DecisionOrder + сайзинг.

Выделено из honey_money_provider.py. Здесь сосредоточены ВСЕ торговые guard-ы,
поэтому их легко ревьюить и тестировать в изоляции:
  • anti-wash-trade — дедуп встречных BUY/SELL по одному тикеру;
  • min-hold — не торгуем тикер чаще, чем раз в N минут;
  • anti-averaging — не усредняем против убыточной позиции;
  • volatility-targeted sizing — equal-risk вес позиций;
  • hard cap — потолок суммы одной сделки от стартового капитала.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.decision.interface import DecisionContext
from src.decision.strategy_params import StrategyParams
from src.schemas import AgentOrder, DecisionAction, DecisionOrder

logger = logging.getLogger(__name__)


class OrderMapper:
    """Переводит AgentOrder (от LLM/fallback) в DecisionOrder с риск-сайзингом."""

    def __init__(self, params: StrategyParams, tickers: List[str]) -> None:
        self.p = params
        self.tickers = tickers

    def map_orders(
        self,
        llm_orders: list,
        context: DecisionContext,
        candles_by_ticker: Dict[str, list],
        expected_vols: Optional[Dict[str, float]] = None,
    ) -> List[DecisionOrder]:
        """Главный метод. Парсит сырые ордера в AgentOrder и прогоняет через guard-ы."""
        expected_vols = expected_vols or {}
        # Типизируем на входе: dict → AgentOrder (непригодные отбрасываем).
        parsed = [ao for ao in (AgentOrder.from_raw(o) for o in llm_orders) if ao is not None]
        orders = self._dedup_wash_trades(parsed)

        cash = float(context.state.cash_balance)
        usable_cash = max(0.0, cash - self.p.min_cash_reserve)
        current_positions = {p.secid: p.position for p in context.positions}
        avg_prices = dict(context.state.average_prices or {})
        last_trade_ts = self._last_trade_times(context)
        now = datetime.now(timezone.utc)

        result: List[DecisionOrder] = []
        for o in orders:
            order = self._map_single(
                o, current_positions, avg_prices, last_trade_ts, now,
                usable_cash, candles_by_ticker, expected_vols,
            )
            if order is not None:
                result.append(order)
        return result

    # ─────────────── Guard helpers ───────────────

    @staticmethod
    def _dedup_wash_trades(orders: List[AgentOrder]) -> List[AgentOrder]:
        """Если для тикера пришли и BUY и SELL — оставляем только первый (анти-wash)."""
        seen: Dict[str, str] = {}
        deduped: List[AgentOrder] = []
        for o in orders:
            action = o.action_upper
            if not o.ticker or action == "HOLD":
                continue
            if o.ticker in seen:
                logger.warning("ANTI-WASH: skip duplicate %s for %s (already have %s)",
                               action, o.ticker, seen[o.ticker])
                continue
            seen[o.ticker] = action
            deduped.append(o)
        return deduped

    @staticmethod
    def _last_trade_times(context: DecisionContext) -> Dict[str, datetime]:
        """Время последней сделки по каждому тикеру (UTC) для min-hold проверки."""
        out: Dict[str, datetime] = {}
        for t in (context.trades_today or []):
            try:
                ts = datetime.fromisoformat(f"{t.tradedate}T{t.tradetime}")
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone(timedelta(hours=3)))  # МСК
                ts_utc = ts.astimezone(timezone.utc)
                if t.secid not in out or ts_utc > out[t.secid]:
                    out[t.secid] = ts_utc
            except Exception:
                continue
        return out

    # ─────────────── Маппинг одного ордера ───────────────

    def _map_single(
        self,
        o: AgentOrder,
        current_positions: Dict[str, int],
        avg_prices: Dict[str, float],
        last_trade_ts: Dict[str, datetime],
        now: datetime,
        usable_cash: float,
        candles_by_ticker: Dict[str, list],
        expected_vols: Dict[str, float],
    ) -> Optional[DecisionOrder]:
        ticker = o.ticker
        our_action = o.action_upper
        # Дефолты берём из StrategyParams, если агент не указал значение.
        size_pct_raw = o.size_pct if o.size_pct is not None else self.p.order_size_pct * 100
        stop_pct_raw = o.stop_loss_pct if o.stop_loss_pct is not None else -self.p.default_stop_pct * 100
        take_pct_raw = o.take_profit_pct if o.take_profit_pct is not None else self.p.default_take_pct * 100
        size_pct = float(size_pct_raw) / 100.0
        stop_pct = float(stop_pct_raw) / 100.0
        take_pct = float(take_pct_raw) / 100.0
        rationale = (o.rationale or "")[:200]

        if ticker not in self.tickers:
            return None

        current_qty = current_positions.get(ticker, 0)
        their_action = self.map_action(our_action, current_qty)
        if their_action is None:
            return None

        ref_price = self._reference_price(ticker, candles_by_ticker)
        if ref_price is None or ref_price <= 0:
            return None

        if self._violates_min_hold(ticker, their_action, last_trade_ts, now):
            return None
        if self._violates_anti_averaging(ticker, their_action, current_qty, ref_price, avg_prices):
            return None

        size_lots, their_action = self._size_order(
            ticker, their_action, current_qty, size_pct, ref_price, usable_cash, expected_vols,
        )
        if size_lots is None:
            return None

        stop_price, take_price = self._stop_take_prices(their_action, current_qty, ref_price, stop_pct, take_pct)

        return DecisionOrder(
            ticker=ticker,
            action=their_action,
            size_lots=size_lots if their_action != DecisionAction.CLOSE else None,
            stop_price=stop_price,
            take_price=take_price,
            priority=100,
            reason_summary=rationale or f"{our_action} from LLM",
        )

    def _violates_min_hold(self, ticker, their_action, last_trade_ts, now) -> bool:
        if ticker not in last_trade_ts:
            return False
        seconds_since = (now - last_trade_ts[ticker]).total_seconds()
        if seconds_since < self.p.min_hold_seconds:
            logger.info("Skip %s %s: last trade %.0fs ago < %ds min-hold",
                        ticker, their_action.value, seconds_since, self.p.min_hold_seconds)
            return True
        return False

    def _violates_anti_averaging(self, ticker, their_action, current_qty, ref_price, avg_prices) -> bool:
        if their_action != DecisionAction.INCREASE:
            return False
        avg_price = float(avg_prices.get(ticker, 0.0))
        if avg_price <= 0:
            return False
        thr = self.p.anti_avg_threshold
        if current_qty > 0 and ref_price < avg_price * (1.0 + thr):
            logger.info("Skip INCREASE long %s: ref %.2f <= avg %.2f (anti-averaging-down)",
                        ticker, ref_price, avg_price)
            return True
        if current_qty < 0 and ref_price > avg_price * (1.0 - thr):
            logger.info("Skip INCREASE short %s: ref %.2f >= avg %.2f (anti-averaging-up)",
                        ticker, ref_price, avg_price)
            return True
        return False

    def _size_order(
        self, ticker, their_action, current_qty, size_pct, ref_price, usable_cash, expected_vols,
    ):
        """Возвращает (size_lots, their_action) либо (None, _) если ордер надо пропустить."""
        opening = (DecisionAction.OPEN_LONG, DecisionAction.INCREASE, DecisionAction.OPEN_SHORT)
        if their_action in opening:
            vol_multiplier = self._vol_multiplier(ticker, expected_vols)
            size_rub = usable_cash * size_pct * vol_multiplier
            if size_rub > self.p.hard_max_order_rub:
                logger.info("Cap order size for %s: %.0f → %.0f ₽ (hard cap)",
                            ticker, size_rub, self.p.hard_max_order_rub)
                size_rub = self.p.hard_max_order_rub
            size_rub = max(size_rub, self.p.min_order_rub)
            size_lots = max(1, int(size_rub / ref_price))
            if size_lots * ref_price > usable_cash * 0.5:
                logger.info("Skip %s %s: 1 lot %.0f > 50%% cash %.0f",
                            ticker, their_action.value, ref_price, usable_cash)
                return None, their_action
            if size_lots * ref_price > self.p.hard_max_order_rub * 1.1:
                size_lots = max(1, int(self.p.hard_max_order_rub / ref_price))
            return size_lots, their_action

        if their_action == DecisionAction.REDUCE:
            size_lots = max(1, abs(current_qty) // 2)
            if size_lots * ref_price < self.p.min_order_rub and abs(current_qty) * ref_price >= self.p.min_order_rub:
                return abs(current_qty), DecisionAction.CLOSE
            return size_lots, their_action

        if their_action == DecisionAction.CLOSE:
            size_lots = abs(current_qty)
            return (size_lots, their_action) if size_lots > 0 else (None, their_action)

        return None, their_action

    def _vol_multiplier(self, ticker: str, expected_vols: Dict[str, float]) -> float:
        ticker_vol = expected_vols.get(ticker)
        if not ticker_vol or ticker_vol <= 0:
            return 1.0
        mult = self.p.target_vol / ticker_vol
        return max(self.p.vol_mult_min, min(self.p.vol_mult_max, mult))

    def _stop_take_prices(self, their_action, current_qty, ref_price, stop_pct, take_pct):
        is_long = their_action == DecisionAction.OPEN_LONG or (
            their_action == DecisionAction.INCREASE and current_qty >= 0)
        is_short = their_action == DecisionAction.OPEN_SHORT or (
            their_action == DecisionAction.INCREASE and current_qty < 0)
        if is_long:
            stop_price = ref_price * (1.0 + stop_pct) if stop_pct < 0 else ref_price * (1.0 - self.p.default_stop_pct)
            take_price = ref_price * (1.0 + take_pct) if take_pct > 0 else ref_price * (1.0 + self.p.default_take_pct)
            return stop_price, take_price
        if is_short:
            return ref_price * (1.0 + self.p.default_stop_pct), ref_price * (1.0 - self.p.default_take_pct)
        return None, None

    @staticmethod
    def map_action(our_action: str, current_qty: int) -> Optional[DecisionAction]:
        """Переводит наш Action → DecisionAction с учётом знака позиции.

        SHORTS включены: SELL_* без позиции открывают/наращивают шорт, что даёт
        боту зарабатывать на падающем рынке (long-only стратегия проигрывает в bear).
        """
        if our_action == "BUY":
            if current_qty < 0:
                return DecisionAction.CLOSE
            return DecisionAction.INCREASE if current_qty > 0 else DecisionAction.OPEN_LONG
        if our_action == "BUY_SMALL":
            if current_qty < 0:
                return DecisionAction.REDUCE
            return DecisionAction.INCREASE if current_qty > 0 else DecisionAction.OPEN_LONG
        if our_action == "SELL_ALL":
            if current_qty > 0:
                return DecisionAction.CLOSE
            if current_qty < 0:
                return DecisionAction.INCREASE
            return DecisionAction.OPEN_SHORT
        if our_action == "SELL_HALF":
            if current_qty > 0:
                return DecisionAction.REDUCE
            if current_qty < 0:
                return None
            return DecisionAction.OPEN_SHORT
        return None

    @staticmethod
    def _reference_price(ticker: str, candles_by_ticker: Dict[str, list]) -> Optional[float]:
        raw = candles_by_ticker.get(ticker) or []
        if not raw:
            return None
        try:
            return float(raw[-1].get("close", 0))
        except (ValueError, TypeError, AttributeError):
            return None
