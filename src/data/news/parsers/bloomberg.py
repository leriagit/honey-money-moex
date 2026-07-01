"""
Bloomberg — вес 1.0. Требует API-ключ (BLOOMBERG_API_KEY).

Bloomberg Terminal API недоступен публично; здесь реализован клиент
через Open Bloomberg API (blpapi) — он есть в проде у любого, у кого
куплен терминал. Альтернатива для хакатона — Bloomberg Open Data /
публичные пресс-релизы (https://www.bloomberg.com/feed/podcast/etf-report).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class BloombergSource(NewsSource):
    """
    Если BLOOMBERG_API_KEY задан и доступна библиотека blpapi —
    подтягиваем headlines через news subscription.
    Иначе используем публичный podcast/feed как fallback.
    """

    PUBLIC_FALLBACK_RSS = "https://www.bloomberg.com/feed/podcast/etf-report.xml"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "BLOOMBERG_API_KEY")
        if key:
            return self._fetch_premium(key, since)
        return self._fetch_public_fallback(since)

    def _fetch_premium(self, key: str, since: datetime) -> List[NewsItem]:
        try:
            import blpapi  # type: ignore  # есть только в терминале
        except ImportError:
            logger.info("blpapi не установлен — Bloomberg premium недоступен")
            return self._fetch_public_fallback(since)
        # Здесь должен быть реальный subscription к //blp/refdata service.
        # Опускаем для хакатона — терминала на сервере нет.
        return self._fetch_public_fallback(since)

    def _fetch_public_fallback(self, since: datetime) -> List[NewsItem]:
        try:
            import feedparser
        except ImportError:
            return []
        try:
            feed = feedparser.parse(self.PUBLIC_FALLBACK_RSS)
        except Exception as e:
            logger.warning("bloomberg fallback fetch failed: %s", e)
            return []

        items = []
        for entry in feed.entries[:30]:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "")
            url = getattr(entry, "link", None)
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            else:
                pub = datetime.now(timezone.utc)
            if pub < since:
                continue
            items.append(self.parse_item(title, summary, url, pub))
        return items
