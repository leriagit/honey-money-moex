"""
RavenPack — вес 0.60. Структурированные sentiment/event-сигналы.
Требует RAVENPACK_API_KEY.
Документация: https://www.ravenpack.com/services/edge/
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class RavenPackSource(NewsSource):
    BASE_URL = "https://api.ravenpack.com/1.0/json/get/rpa-cme-news"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "RAVENPACK_API_KEY")
        if not key:
            return []
        try:
            import requests
        except ImportError:
            return []
        params = {
            "fields": "TIMESTAMP_UTC,RP_ENTITY_ID,ENTITY_NAME,EVENT_SENTIMENT_SCORE,EVENT_RELEVANCE,HEADLINE",
            "start_date": since.strftime("%Y-%m-%dT%H:%M:%S"),
            "limit": 200,
        }
        try:
            r = requests.get(
                self.BASE_URL,
                params=params,
                headers={"API_KEY": key},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("RavenPack fetch failed: %s", e)
            return []

        items = []
        for row in data.get("records", []):
            try:
                pub = datetime.fromisoformat(row["TIMESTAMP_UTC"].replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)
            sentiment = float(row.get("EVENT_SENTIMENT_SCORE", 0.0))
            relevance = float(row.get("EVENT_RELEVANCE", 0.0)) / 100.0
            entity = row.get("ENTITY_NAME", "")
            items.append(self.parse_item(
                title=row.get("HEADLINE", ""),
                summary=f"Entity: {entity}",
                url=None,
                published_at=pub,
                explicit_sentiment=sentiment,
                explicit_tickers=[entity] if entity else None,
                raw_score=relevance,
            ))
        return items
