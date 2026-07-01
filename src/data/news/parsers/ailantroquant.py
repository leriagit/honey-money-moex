"""
AilantroQuant — Telegram-канал. Вес 0.60.
Требует TELEGRAM_API_ID + TELEGRAM_API_HASH + TELEGRAM_SESSION (telethon).

В прод-режиме читаем последние посты канала через MTProto.
В fallback-режиме (нет ключей) — пусто.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


CHANNEL = "AilantroQuant"


class AilantroQuantSource(NewsSource):
    def fetch(self, since: datetime) -> List[NewsItem]:
        api_id = os.environ.get("TELEGRAM_API_ID")
        api_hash = os.environ.get("TELEGRAM_API_HASH")
        session = os.environ.get("TELEGRAM_SESSION", "honey_money_session")
        if not api_id or not api_hash:
            return []

        try:
            from telethon.sync import TelegramClient  # type: ignore
        except ImportError:
            logger.debug("telethon не установлен — AilantroQuant пропущен (опционально)")
            return []

        items: List[NewsItem] = []
        try:
            with TelegramClient(session, int(api_id), api_hash) as client:
                for msg in client.iter_messages(CHANNEL, limit=50):
                    if not msg.message:
                        continue
                    pub = msg.date.astimezone(timezone.utc) if msg.date else datetime.now(timezone.utc)
                    if pub < since:
                        break
                    items.append(self.parse_item(
                        title=msg.message[:120],
                        summary=msg.message,
                        url=f"https://t.me/{CHANNEL}/{msg.id}",
                        published_at=pub,
                    ))
        except Exception as e:
            logger.warning("AilantroQuant fetch failed: %s", e)
        return items
