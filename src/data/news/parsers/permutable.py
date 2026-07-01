"""
Permutable AI — вес 0.85. Многоязычный новостной AI-аналитик (37 языков).
Требует PERMUTABLE_API_KEY.
Документация: https://permutable.ai/api/
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class PermutableSource(NewsSource):
    BASE_URL = "https://api.permutable.ai/v1/signals"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "PERMUTABLE_API_KEY")
        if not key:
            return []
        try:
            import requests
        except ImportError:
            return []
        try:
            r = requests.get(
                self.BASE_URL,
                params={"since": since.isoformat()},
                headers={"X-API-Key": key},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Permutable fetch failed: %s", e)
            return []

        items = []
        for row in data.get("signals", []):
            try:
                pub = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)
            sentiment = float(row.get("sentiment", 0.0))
            asset = row.get("asset", "")
            items.append(self.parse_item(
                title=row.get("headline", ""),
                summary=row.get("rationale", ""),
                url=row.get("source_url"),
                published_at=pub,
                explicit_sentiment=sentiment,
                explicit_tickers=[asset] if asset else None,
                raw_score=row.get("confidence"),
            ))
        return items
