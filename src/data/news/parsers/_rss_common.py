"""
Общий RSS-парсер. Используется всеми источниками с rss_url:
Reuters, Investing, NYT, WSJ, FT, FoxNews, IEA, WaPo.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class RSSSource(NewsSource):
    """Базовая реализация для любого источника с RSS-фидом."""

    def fetch(self, since: datetime) -> List[NewsItem]:
        if not self.cfg.rss_url:
            return []
        try:
            import feedparser  # отложенный импорт, чтобы не падал тест на схемах
        except ImportError:
            logger.warning("feedparser not installed — RSS источник %s пропущен", self.id)
            return []

        try:
            feed = feedparser.parse(self.cfg.rss_url)
        except Exception as e:
            logger.warning("RSS fetch failed for %s: %s", self.id, e)
            return []

        items: List[NewsItem] = []
        for entry in feed.entries[:50]:
            try:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                url = getattr(entry, "link", None)
                # Парсим дату
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                else:
                    pub = datetime.now(timezone.utc)

                if pub < since:
                    continue

                items.append(self.parse_item(
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=pub,
                ))
            except Exception as e:
                logger.debug("rss entry parse error in %s: %s", self.id, e)
                continue
        return items
