"""
Общая база для всех Telegram-источников.

Парсим через telethon, user account. См. docs/DATA_ACCESS.md →
"Категория A. Telegram-каналы".

Все конкретные ТГК — короткие наследники TelegramChannelSource:
просто переопределяют CHANNEL и (опционально) extract_tickers().
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class TelegramChannelSource(NewsSource):
    """
    Источник новостей из публичного Telegram-канала.

    Наследник минимум переопределяет CHANNEL (без @).
    Опционально — sentiment_hint() для канала-специфичной обработки.
    """

    # Username канала без @. ПЕРЕОПРЕДЕЛЯЕМ В НАСЛЕДНИКЕ.
    CHANNEL: str = ""

    # Лимит сообщений за один fetch
    MESSAGE_LIMIT: int = 50

    def fetch(self, since: datetime) -> List[NewsItem]:
        if not self.CHANNEL:
            return []

        api_id = os.environ.get("TELEGRAM_API_ID")
        api_hash = os.environ.get("TELEGRAM_API_HASH")
        session = os.environ.get("TELEGRAM_SESSION", "honey_money_session")

        if not api_id or not api_hash:
            logger.debug("TELEGRAM_API_ID/HASH не заданы — %s пропущен", self.id)
            return []

        try:
            from telethon.sync import TelegramClient  # type: ignore
            from telethon.errors import FloodWaitError  # type: ignore
        except ImportError:
            logger.debug("telethon не установлен — %s пропущен (опционально)", self.id)
            return []

        items: List[NewsItem] = []
        try:
            with TelegramClient(session, int(api_id), api_hash) as client:
                for msg in client.iter_messages(self.CHANNEL, limit=self.MESSAGE_LIMIT):
                    text = msg.message or ""
                    if not text.strip():
                        continue
                    pub = msg.date.astimezone(timezone.utc) if msg.date else datetime.now(timezone.utc)
                    if pub < since:
                        break
                    item = self.parse_item(
                        title=text[:120],
                        summary=text,
                        url=f"https://t.me/{self.CHANNEL}/{msg.id}",
                        published_at=pub,
                        explicit_sentiment=self.sentiment_hint(text),
                    )
                    # apply per-source sentiment override
                    items.append(item)
        except FloodWaitError as e:
            logger.warning("FloodWait %ss for %s, skip this cycle", e.seconds, self.id)
        except Exception as e:
            logger.warning("Telegram fetch failed for %s: %s", self.id, e)
        return items

    # ---- хук для наследников: канал-специфичный sentiment ----
    def sentiment_hint(self, text: str) -> Optional[float]:
        """
        Может вернуть конкретный sentiment для этого канала.
        По умолчанию None → base.parse_item использует heuristic_sentiment.

        Полезно для каналов типа Дивиденды Онлайн (нейтральный агрегатор) —
        всегда возвращаем 0.0, sentiment не имеет смысла.
        """
        return None
