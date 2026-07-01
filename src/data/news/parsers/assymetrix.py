"""
Assymetrix — вес 0.55. Глубокие on-chain и prediction-markets данные.
Требует ASSYMETRIX_API_KEY.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class AssymetrixSource(NewsSource):
    BASE_URL = "https://api.assymetrix.com/v1/markets"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "ASSYMETRIX_API_KEY")
        if not key:
            return []
        try:
            import requests
        except ImportError:
            return []
        try:
            r = requests.get(
                self.BASE_URL,
                headers={"Authorization": f"Bearer {key}"},
                params={"updated_since": since.isoformat()},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Assymetrix fetch failed: %s", e)
            return []

        items = []
        for row in data.get("markets", []):
            try:
                pub = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)
            # Prediction market: probability shift trades as sentiment
            shift = float(row.get("probability_shift", 0.0))   # [-1, 1]
            items.append(self.parse_item(
                title=row.get("question", ""),
                summary=row.get("category", ""),
                url=row.get("url"),
                published_at=pub,
                explicit_sentiment=max(-1.0, min(1.0, shift)),
                raw_score=row.get("volume_usd"),
            ))
        return items
