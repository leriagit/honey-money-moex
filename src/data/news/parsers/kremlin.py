"""
Кремль / kremlin.ru — геополитика и официальные указы Президента.
Вес 1.00 (любое заявление = 3-8% движение IMOEX в моменте).

Помечаем news.regulator_event = True если в новости есть:
- "Указ Президента", "мобилизация", "военное положение"
- "санкции", "СВО", "Украина"
- "ВПК", "выпуск ОФЗ", "национализация"

RSS публичный, без ключа.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


KREMLIN_TRIGGERS = [
    "указ президента", "мобилизац", "военное положение",
    "новых санкций", "санкции против", "контрсанкц",
    "национализац", "конфискация", "выпуск офз",
    "военно-промышленн", "указ о", "ввп",
]


class KremlinSource(NewsSource):
    RSS_URL = "http://kremlin.ru/events/all/feed"

    def fetch(self, since: datetime) -> List[NewsItem]:
        try:
            import feedparser
        except ImportError:
            return []
        try:
            feed = feedparser.parse(self.RSS_URL)
        except Exception as e:
            logger.warning("Kremlin RSS failed: %s", e)
            return []

        items: List[NewsItem] = []
        for entry in feed.entries[:40]:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            url = getattr(entry, "link", None)
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            else:
                pub = datetime.now(timezone.utc)
            if pub < since:
                continue
            items.append(self.parse_item(
                title=title, summary=summary, url=url,
                published_at=pub,
                explicit_ru=1.0,
            ))
        return items
