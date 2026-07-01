"""
Банк России (ЦБ РФ) — официальный регулятор, вес 1.00.

Решения по ключевой ставке двигают весь рынок мгновенно.
Источник: публичный RSS на cbr.ru — никаких ключей не нужно.

Помечаем news.regulator_event = True если в заголовке упоминается
"ключевая ставка", "ставка", "денежно-кредитная политика" — это
триггер для circuit-breaker логики в news_aggregator.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


# Ключевые триггеры для регуляторных событий
CBR_KEY_TRIGGERS = [
    "ключевая ставка", "ключевой ставки", "ключевую ставку",
    "денежно-кредитной политики", "ДКП", "решение совета директоров",
    "процентная ставка",
]


class CBRSource(NewsSource):
    """Парсит ленту cbr.ru/rss/eventrss (события + пресс-релизы)."""

    RSS_URLS = [
        "https://www.cbr.ru/rss/eventrss",       # события (заседания и т.п.)
        "https://www.cbr.ru/rss/RssPress",        # пресс-релизы
    ]

    def fetch(self, since: datetime) -> List[NewsItem]:
        try:
            import feedparser
        except ImportError:
            return []

        items: List[NewsItem] = []
        for url in self.RSS_URLS:
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                logger.warning("CBR fetch %s failed: %s", url, e)
                continue
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                url_entry = getattr(entry, "link", None)
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                else:
                    pub = datetime.now(timezone.utc)
                if pub < since:
                    continue
                # Все ЦБ-новости заведомо РУ-релевантны
                item = self.parse_item(
                    title=title, summary=summary, url=url_entry,
                    published_at=pub,
                    explicit_ru=1.0,
                )
                items.append(item)
        return items
