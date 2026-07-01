"""Часовые/дневные свечи MOEX ISS для compute_indicators и ML-стека.

Расширение к существующему MoexISSMarketDataProvider, который умеет только
get_last_price. Здесь — батч-запрос OHLCV для всех 20 тикеров за окно.

Используется в HoneyMoneyDecisionProvider для построения indicators
и подачи в LightGBM. На каждый decision_cycle (раз в DECISION_INTERVAL_SECONDS)
делаем 20 параллельных запросов к ISS — ~5-10 секунд.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MoexCandlesProvider:
    """Получает часовые OHLCV свечи по списку тикеров через aiomoex."""

    def __init__(self, lookback_hours: int = 120) -> None:
        """
        Args:
            lookback_hours: сколько часов истории качать (по умолчанию 120 = 5 дней)
        """
        self.lookback_hours = lookback_hours

    async def _fetch_one(self, session, ticker: str, start: str, end: str) -> List[dict]:
        """Часовые свечи для одного тикера."""
        try:
            import aiomoex
            data = await aiomoex.get_board_candles(
                session, security=ticker, interval=60, start=start, end=end,
            )
            return data or []
        except Exception as e:
            logger.warning("MoexCandles: failed for %s: %s", ticker, e)
            return []

    async def _fetch_all_async(self, tickers: List[str]) -> Dict[str, List[dict]]:
        """Параллельно запрашивает свечи для всех тикеров."""
        import aiohttp
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=self.lookback_hours)
        start = start_dt.date().isoformat()
        end = end_dt.date().isoformat()

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self._fetch_one(session, tk, start, end) for tk in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        return {tk: data for tk, data in zip(tickers, results)}

    def fetch_all(self, tickers: List[str]) -> Dict[str, List[dict]]:
        """Sync обёртка над async-фетчем. Возвращает {ticker: [{ts, open, high, low, close, volume}]}."""
        try:
            return asyncio.run(self._fetch_all_async(tickers))
        except RuntimeError:
            # Уже в running loop — создаём новый
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._fetch_all_async(tickers))
            finally:
                loop.close()
