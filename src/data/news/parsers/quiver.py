"""
Quiver Quantitative — вес 0.30.
Источники: Конгресс-трейды, инсайдеры, Reddit sentiment, гос.контракты.
Требует QUIVER_API_KEY.
Документация: https://api.quiverquant.com/
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class QuiverSource(NewsSource):
    BASE_URL = "https://api.quiverquant.com/beta/live"

    def fetch(self, since: datetime) -> List[NewsItem]:
        key = os.environ.get(self.cfg.env_key or "QUIVER_API_KEY")
        if not key:
            return []
        try:
            import requests
        except ImportError:
            return []
        items: List[NewsItem] = []
        endpoints = [
            ("congresstrading", "Congress trade"),
            ("insiders", "Insider activity"),
            ("wallstreetbets", "Reddit WSB sentiment"),
            ("govcontractsall", "Government contract"),
        ]
        headers = {"Authorization": f"Bearer {key}"}
        for ep, label in endpoints:
            try:
                r = requests.get(f"{self.BASE_URL}/{ep}", headers=headers, timeout=10)
                r.raise_for_status()
                rows = r.json()
            except Exception as e:
                logger.warning("Quiver %s failed: %s", ep, e)
                continue
            for row in (rows or [])[:30]:
                try:
                    pub_raw = row.get("Date") or row.get("ReportDate") or row.get("Time")
                    pub = datetime.fromisoformat(pub_raw) if pub_raw else datetime.now(timezone.utc)
                except Exception:
                    pub = datetime.now(timezone.utc)
                ticker = row.get("Ticker", "")
                tx_type = row.get("Transaction", "")
                # Insiders/Congress: Purchase → +0.4, Sale → -0.4
                sentiment = 0.0
                if "purchase" in tx_type.lower() or "buy" in tx_type.lower():
                    sentiment = 0.4
                elif "sale" in tx_type.lower() or "sell" in tx_type.lower():
                    sentiment = -0.4
                # WSB: используем Sentiment поле напрямую
                if "Sentiment" in row:
                    try:
                        sentiment = max(-1.0, min(1.0, float(row["Sentiment"])))
                    except Exception:
                        pass
                items.append(self.parse_item(
                    title=f"{label}: {ticker} {tx_type}",
                    summary=str(row),
                    url=None,
                    published_at=pub,
                    explicit_sentiment=sentiment,
                    explicit_tickers=[ticker] if ticker else None,
                ))
        return items
