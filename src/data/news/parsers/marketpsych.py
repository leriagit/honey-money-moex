"""
MarketPsych — вес 0.90. AI/NLP-сигналы из новостей и соцсетей.
Требует MARKETPSYCH_API_KEY (или RefinitivQA).

Документация эндпоинтов: https://www.marketpsych.com/data/
Формат: JSON Lines с полями `assetCode`, `buzz`, `sentiment`, `urgency`.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class MarketPsychSource(NewsSource):
    BASE_URL = "https://api.marketpsych.com/v1/feed"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "MARKETPSYCH_API_KEY")
        if not key:
            logger.info("MARKETPSYCH_API_KEY не задан — пропускаем источник")
            return []

        try:
            import requests
        except ImportError:
            return []

        params = {
            "since": int(since.timestamp()),
            "format": "json",
            "asset_class": "EQ",
        }
        headers = {"Authorization": f"Bearer {key}"}
        try:
            r = requests.get(self.BASE_URL, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("MarketPsych fetch failed: %s", e)
            return []

        items = []
        for row in data.get("items", []):
            ts = row.get("timestamp")
            pub = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
            sentiment_score = float(row.get("sentiment", 0.0))   # уже [-1, 1]
            asset = row.get("assetCode", "")
            items.append(self.parse_item(
                title=row.get("headline", ""),
                summary=row.get("snippet", ""),
                url=row.get("url"),
                published_at=pub,
                explicit_sentiment=sentiment_score,
                explicit_tickers=[asset] if asset else None,
                raw_score=row.get("buzz"),
            ))
        return items
