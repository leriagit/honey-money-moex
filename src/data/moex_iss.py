"""Последняя цена инструмента через MOEX ISS.

Robustness:
  - Большие timeout'ы (20 connect / 30 read) — iss.moex.com бывает медленный.
  - In-memory кеш с TTL: при ошибке отдаём последнюю известную цену моложе TTL.
  - Логируем timeout на DEBUG (норма для деградированного режима), а не ERROR.

Это критично для stop_watcher: при недоступности ISS API стопы не должны
"замерзать" — лучше отдать чуть устаревшую цену, чем None и пропуск проверки.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MoexISSMarketDataProvider:
    """HTTP-адаптер к публичному MOEX ISS с кешем и stale-fallback."""

    def __init__(
        self,
        base_url: str = "https://iss.moex.com/iss",
        timeout_seconds: float = 30.0,        # было 10s — слишком мало для нестабильного ISS
        connect_timeout: float = 20.0,
        max_retries: int = 3,
        cache_ttl_seconds: float = 600.0,     # 10 минут — допустимая stale при отказе ISS
    ):
        """Создает HTTP-клиент ISS с timeout, retry и кешем."""

        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout)
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        # ticker → (price, unix_ts)
        self._price_cache: Dict[str, Tuple[float, float]] = {}

    def close(self) -> None:
        """Закрывает HTTP-клиент ISS."""

        self._client.close()

    # ─── public API ────────────────────────────────────────────────────

    def get_last_price(self, secid: str) -> float | None:
        """
        Возвращает последнюю цену акции через MOEX ISS.

        Если ISS недоступен — отдаёт stale-цену из кеша (если она моложе TTL).
        Возвращает None только если ни свежей, ни stale-цены нет.
        """
        # 1) пробуем сходить в ISS
        try:
            price = self._fetch_with_retry(secid)
            if price is not None:
                self._price_cache[secid] = (price, time.time())
                return price
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as e:
            # ожидаемый сценарий деградации — на DEBUG, не ERROR (не флудим логи)
            logger.debug("MOEX ISS unavailable for %s: %s", secid, type(e).__name__)
        except Exception as e:  # noqa: BLE001
            logger.warning("MOEX ISS unexpected error for %s: %s", secid, e)

        # 2) ISS не дал ответа — пробуем stale из кеша
        cached = self._price_cache.get(secid)
        if cached is None:
            return None
        price, ts = cached
        age = time.time() - ts
        if age > self.cache_ttl_seconds:
            logger.debug("MOEX ISS cache stale for %s (age=%.0fs)", secid, age)
            return None
        logger.debug("MOEX ISS using cached price for %s (age=%.0fs)", secid, age)
        return price

    # ─── internals ─────────────────────────────────────────────────────

    def _fetch_with_retry(self, secid: str) -> float | None:
        """Запрос с tenacity-ретраем. Бросает исключение при провале всех попыток."""
        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, max=5),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            reraise=True,
        )
        data = None
        for attempt in retryer:
            with attempt:
                response = self._client.get(
                    f"/engines/stock/markets/shares/securities/{secid}.json",
                    params={
                        "iss.meta": "off",
                        "iss.only": "marketdata",
                        "marketdata.columns": "SECID,LAST,MARKETPRICE,LCURRENTPRICE",
                    },
                )
                response.raise_for_status()
                data = response.json()

        if data is None:
            return None

        marketdata = data.get("marketdata", {})
        columns = marketdata.get("columns", [])
        rows = marketdata.get("data", [])
        if not rows:
            return None

        row = dict(zip(columns, rows[0], strict=False))
        for field in ("LAST", "MARKETPRICE", "LCURRENTPRICE"):
            value = row.get(field)
            if value is not None:
                return float(value)
        return None

    # ─── для интеграции с decision-провайдером ─────────────────────────

    def prime_cache(self, secid: str, price: float) -> None:
        """
        Обновить кеш извне (например, из decision-провайдера, которая
        качает candles через aiomoex). Это позволяет stop_watcher
        получать свежую цену даже если REST-ISS лежит.
        """
        if price > 0:
            self._price_cache[secid] = (float(price), time.time())
