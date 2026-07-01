"""
Alpha Vantage — вес 0.30. Stocks/forex/crypto + новости с готовым sentiment.
Требует ALPHAVANTAGE_API_KEY.
Документация: https://www.alphavantage.co/documentation/#news-sentiment
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class AlphaVantageSource(NewsSource):
    BASE_URL = "https://www.alphavantage.co/query"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "ALPHAVANTAGE_API_KEY")
        if not key:
            return []
        try:
            import requests
        except ImportError:
            return []
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "financial_markets,economy_macro,energy_transportation",
            "time_from": since.strftime("%Y%m%dT%H%M"),
            "limit": 200,
            "apikey": key,
        }
        try:
            r = requests.get(self.BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("AlphaVantage fetch failed: %s", e)
            return []

        items = []
        for row in data.get("feed", []):
            try:
                pub = datetime.strptime(row["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                pub = datetime.now(timezone.utc)
            # overall_sentiment_score уже в [-1, 1]
            sentiment = float(row.get("overall_sentiment_score", 0.0))
            tickers = [t["ticker"] for t in row.get("ticker_sentiment", [])]
            items.append(self.parse_item(
                title=row.get("title", ""),
                summary=row.get("summary", ""),
                url=row.get("url"),
                published_at=pub,
                explicit_sentiment=sentiment,
                explicit_tickers=tickers or None,
                raw_score=row.get("overall_sentiment_label"),
            ))
        return items
