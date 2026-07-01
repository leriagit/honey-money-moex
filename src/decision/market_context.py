"""Сбор внешних рыночных данных для decision-провайдера.

Выделено из honey_money_provider.py (god-object на 936 строк). Здесь живёт всё,
что ходит в сеть: часовые свечи MOEX ISS, RSS-новости и дневной IMOEX. Каждый
источник имеет собственный TTL-кэш и жёсткий таймаут, чтобы сетевые проблемы
никогда не блокировали торговый цикл.

Раньше методы _fetch_news_safe и _fetch_imoex_returns_safe были СЛУЧАЙНО
определены в провайдере дважды — вторая версия молча перекрывала первую
(~100 строк мёртвого кода). Здесь — по одной канонической реализации.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Канонические коды инструментов MOEX ISS для макро-контекста.
USD_RUB_SECID = "USD000UTSTOM"   # USD/RUB TOM, валютный рынок, борд CETS
BRENT_ASSET_CODE = "BR"          # фьючерсы Brent на FORTS (ASSETCODE=BR)
_ISS_BASE = "https://iss.moex.com/iss"
_ISS_TIMEOUT_SEC = 15.0


class MarketContextFetcher:
    """Кэширующий загрузчик свечей, новостей, IMOEX и макро-котировок MOEX."""

    def __init__(
        self,
        tickers: List[str],
        candles_provider: Optional[Any],
        news_aggregator: Optional[Any],
        market_data: Optional[Any] = None,
        news_fetch_interval_sec: float = 1800.0,
        imoex_fetch_interval_sec: float = 3600.0,
        news_fetch_timeout_sec: float = 30.0,
        macro_fetch_interval_sec: float = 3600.0,
    ) -> None:
        self.tickers = tickers
        self._candles = candles_provider
        self._news_aggregator = news_aggregator
        self._market_data = market_data
        self._news_fetch_interval_sec = news_fetch_interval_sec
        self._imoex_fetch_interval_sec = imoex_fetch_interval_sec
        self._news_fetch_timeout_sec = news_fetch_timeout_sec
        self._macro_fetch_interval_sec = macro_fetch_interval_sec

        self._news_cache: List[Dict[str, Any]] = []
        self._last_news_fetch_ts: float = 0.0
        self._imoex_returns_cache: Dict[str, float] = {"5d": 0.0, "1d": 0.0}
        self._last_imoex_fetch_ts: float = 0.0
        # Кэши живых макро-котировок MOEX (None = ещё не получали / недоступно).
        self._usd_rub_cache: Optional[float] = None
        self._last_usd_rub_ts: float = 0.0
        self._brent_cache: Optional[float] = None
        self._last_brent_ts: float = 0.0

    # ─────────────── Свечи ───────────────

    def fetch_candles(self) -> Dict[str, list]:
        """Возвращает {ticker: [candle_dicts]}. При ошибке — пустой dict.

        Side-effect: после успешного fetch обновляет кеш цен в market_data —
        это позволяет stop_watcher продолжать работу даже при отказе ISS REST.
        """
        if self._candles is None:
            return {}
        try:
            result = self._candles.fetch_all(self.tickers)
        except Exception as e:
            logger.warning("Candles fetch failed: %s", e)
            return {}

        self._prime_price_cache(result)
        return result

    def _prime_price_cache(self, candles_by_ticker: Dict[str, list]) -> None:
        if self._market_data is None or not hasattr(self._market_data, "prime_cache"):
            return
        for ticker, candles in candles_by_ticker.items():
            if not candles:
                continue
            last = candles[-1]
            close = last.get("close") if isinstance(last, dict) else getattr(last, "close", None)
            if close is not None and close > 0:
                try:
                    self._market_data.prime_cache(ticker, float(close))
                except Exception:
                    pass  # никогда не валим decision-цикл из-за кеша

    # ─────────────── Новости ───────────────

    def fetch_news(self) -> List[Dict[str, Any]]:
        """Возвращает заголовки [{title, source, ru_relevance, sentiment}].

        Кэш на news_fetch_interval_sec. Сам collect() выполняется в отдельном
        потоке с таймаутом — feedparser по 37 источникам не должен подвешивать цикл.
        """
        if self._news_aggregator is None:
            return []

        now = time.time()
        if self._news_cache and (now - self._last_news_fetch_ts) < self._news_fetch_interval_sec:
            return self._news_cache

        result_box: Dict[str, Any] = {"items": None, "error": None}

        def _do_fetch() -> None:
            try:
                result_box["items"] = self._news_aggregator.collect()
            except Exception as e:  # noqa: BLE001 — фоновый поток, логируем и продолжаем
                result_box["error"] = str(e)

        thread = threading.Thread(target=_do_fetch, daemon=True)
        thread.start()
        thread.join(timeout=self._news_fetch_timeout_sec)

        if thread.is_alive():
            logger.warning("News fetch timed out — using cached headlines (%d items)",
                           len(self._news_cache))
            return self._news_cache
        if result_box["error"]:
            logger.warning("News fetch failed: %s", result_box["error"])
            return self._news_cache

        items = result_box["items"] or []
        try:
            items_sorted = sorted(items, key=lambda x: x.published_at, reverse=True)[:30]
        except Exception:
            items_sorted = items[:30]

        headlines = [
            {
                "title": item.title,
                "source": item.source_id,
                "ru_relevance": item.ru_relevance,
                "sentiment": item.sentiment,
            }
            for item in items_sorted
        ]
        self._news_cache = headlines
        self._last_news_fetch_ts = now
        logger.info("News fetched: %d headlines (next refresh in %d sec)",
                    len(headlines), int(self._news_fetch_interval_sec))
        return headlines

    # ─────────────── IMOEX индекс ───────────────

    def fetch_imoex_returns(self) -> Dict[str, float]:
        """Возвращает {"5d": pct, "1d": pct, "last_close": ...} IMOEX. Кэш на час."""
        now = time.time()
        if (now - self._last_imoex_fetch_ts) < self._imoex_fetch_interval_sec:
            return self._imoex_returns_cache

        data = self._fetch_imoex_candles()
        if not data or len(data) < 2:
            return self._imoex_returns_cache

        closes = [float(d.get("close", 0)) for d in data if d.get("close")]
        if len(closes) < 2:
            return self._imoex_returns_cache

        last = closes[-1]
        day_before = closes[-2]
        five_days_before = closes[-6] if len(closes) >= 6 else closes[0]
        self._imoex_returns_cache = {
            "1d": (last / day_before - 1.0) if day_before > 0 else 0.0,
            "5d": (last / five_days_before - 1.0) if five_days_before > 0 else 0.0,
            "last_close": last,
            "realized_vol": self.realized_vol(closes),
        }
        self._last_imoex_fetch_ts = now
        logger.info("IMOEX returns updated: 1d=%.3f%%, 5d=%.3f%%, vol=%.3f%%",
                    self._imoex_returns_cache["1d"] * 100,
                    self._imoex_returns_cache["5d"] * 100,
                    self._imoex_returns_cache["realized_vol"] * 100)
        return self._imoex_returns_cache

    @staticmethod
    def realized_vol(closes: List[float]) -> float:
        """Дневная realized volatility = выборочный std дневных доходностей.

        Меньше двух доходностей → 0.0 (нечего считать).
        """
        rets = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return var ** 0.5

    # ─────────── Живые макро-котировки MOEX (USD/RUB, Brent) ───────────

    def fetch_usd_rub(self) -> Optional[float]:
        """Последний дневной close USD/RUB (CETS). None → MOEX недоступен."""
        now = time.time()
        if self._usd_rub_cache is not None and (now - self._last_usd_rub_ts) < self._macro_fetch_interval_sec:
            return self._usd_rub_cache
        price = self._fetch_currency_last_close(USD_RUB_SECID)
        if price is not None and price > 0:
            self._usd_rub_cache = price
            self._last_usd_rub_ts = now
            logger.info("USD/RUB from MOEX: %.4f", price)
        return self._usd_rub_cache

    def fetch_brent_usd(self) -> Optional[float]:
        """LAST по ближайшему фьючерсу Brent (FORTS). None → недоступно."""
        now = time.time()
        if self._brent_cache is not None and (now - self._last_brent_ts) < self._macro_fetch_interval_sec:
            return self._brent_cache
        price = self._fetch_brent_front_price()
        if price is not None and price > 0:
            self._brent_cache = price
            self._last_brent_ts = now
            logger.info("Brent from MOEX: %.2f", price)
        return self._brent_cache

    @classmethod
    def _fetch_currency_last_close(cls, secid: str) -> Optional[float]:
        """Последний дневной close инструмента валютного рынка (борд CETS)."""
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=10)
        path = f"/engines/currency/markets/selt/boards/CETS/securities/{secid}/candles.json"
        data = cls._iss_get(path, {"interval": "24", "from": start.isoformat(),
                                   "till": end.isoformat(), "iss.meta": "off"})
        return cls._last_candle_close(data)

    @classmethod
    def _fetch_brent_front_price(cls) -> Optional[float]:
        """Находит ближайший незавершённый фьючерс Brent и берёт его LAST."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        listing = cls._iss_get(
            "/engines/futures/markets/forts/securities.json",
            {"iss.meta": "off", "iss.only": "securities",
             "securities.columns": "SECID,ASSETCODE,LASTTRADEDATE"},
        )
        secid = cls._pick_front_future(listing, BRENT_ASSET_CODE, today)
        if secid is None:
            return None
        md = cls._iss_get(
            f"/engines/futures/markets/forts/securities/{secid}.json",
            {"iss.meta": "off", "iss.only": "marketdata",
             "marketdata.columns": "SECID,LAST,LCURRENTPRICE"},
        )
        return cls._marketdata_price(md)

    # ─────────── ISS низкоуровневые помощники (чистые/тестируемые) ───────────

    @staticmethod
    def _iss_get(path: str, params: Dict[str, str]) -> Optional[dict]:
        """GET к ISS REST → распарсенный JSON или None при любой ошибке."""
        try:
            import aiohttp

            async def _fetch() -> Optional[dict]:
                timeout = aiohttp.ClientTimeout(total=_ISS_TIMEOUT_SEC)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(_ISS_BASE + path, params=params) as resp:
                        if resp.status != 200:
                            return None
                        return await resp.json(content_type=None)

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_fetch())
            finally:
                loop.close()
        except Exception as e:
            logger.warning("ISS GET %s failed: %s", path, e)
            return None

    @staticmethod
    def _rows(data: Optional[dict], block: str) -> List[dict]:
        """ISS-блок {columns, data} → список dict-строк."""
        if not data or block not in data:
            return []
        section = data[block]
        cols = section.get("columns") or []
        return [dict(zip(cols, row)) for row in (section.get("data") or [])]

    @classmethod
    def _last_candle_close(cls, data: Optional[dict]) -> Optional[float]:
        for row in reversed(cls._rows(data, "candles")):
            close = row.get("close")
            if close is not None:
                try:
                    return float(close)
                except (TypeError, ValueError):
                    continue
        return None

    @classmethod
    def _pick_front_future(cls, data: Optional[dict], asset_code: str,
                           today_iso: str) -> Optional[str]:
        """SECID ближайшего по экспирации незавершённого фьючерса с данным ASSETCODE."""
        candidates = []
        for row in cls._rows(data, "securities"):
            if row.get("ASSETCODE") != asset_code:
                continue
            ltd, secid = row.get("LASTTRADEDATE"), row.get("SECID")
            if secid and ltd and ltd >= today_iso:
                candidates.append((ltd, secid))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    @classmethod
    def _marketdata_price(cls, data: Optional[dict]) -> Optional[float]:
        for row in cls._rows(data, "marketdata"):
            for key in ("LAST", "LCURRENTPRICE"):
                val = row.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _fetch_imoex_candles() -> list:
        """Тянет дневные свечи IMOEX через aiomoex. Любая ошибка → []."""
        try:
            import aiohttp
            import aiomoex
            from datetime import datetime, timedelta, timezone

            async def _fetch() -> list:
                end = datetime.now(timezone.utc).date()
                start = end - timedelta(days=15)
                timeout = aiohttp.ClientTimeout(total=15)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    data = await aiomoex.get_board_candles(
                        session, security="IMOEX", interval=24,
                        start=start.isoformat(), end=end.isoformat(),
                        board="SNDX", market="index", engine="stock",
                    )
                return data or []

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_fetch())
            finally:
                loop.close()
        except Exception as e:
            logger.warning("IMOEX fetch failed: %s", e)
            return []
