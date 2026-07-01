"""
Московская биржа — официальные новости (листинг/делистинг, изменения в индексах).
Вес 1.00. Никаких ключей не нужно.

ВАЖНО: это новости MOEX, а не котировки. Котировки идут отдельно через
aiomoex ISS API в src/data/market_data.py (вне этого модуля).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class MoexNewsSource(NewsSource):
    """Парсит ISS news endpoint MOEX."""

    BASE_URL = "https://iss.moex.com/iss/sitenews.json"

    def fetch(self, since: datetime) -> List[NewsItem]:
        try:
            import requests
        except ImportError:
            return []
        try:
            r = requests.get(self.BASE_URL, params={"limit": 50}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("MOEX news fetch failed: %s", e)
            return []

        items = []
        block = data.get("sitenews", {})
        columns = block.get("columns", [])
        rows = block.get("data", [])
        idx = {c: i for i, c in enumerate(columns)}
        for row in rows:
            try:
                pub_raw = row[idx.get("modified_at", idx.get("published_at", 0))]
                pub = datetime.strptime(pub_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                pub = datetime.now(timezone.utc)
            if pub < since:
                continue
            title = row[idx.get("title", 0)] if "title" in idx else ""
            body  = row[idx.get("body", 0)] if "body" in idx else ""
            url   = row[idx.get("url", 0)] if "url" in idx else None
            items.append(self.parse_item(
                title=title, summary=body, url=url,
                published_at=pub,
                explicit_ru=1.0,
            ))
        return items
