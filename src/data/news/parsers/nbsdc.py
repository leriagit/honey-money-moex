"""
NBSDC — 金融事件级情感分析数据集. Вес 0.75.
Event-level разметка: тикеры, отрасль, тип событий, эмоциональная популярность.

Источник китайский, поэтому полезен скорее как калибровочный датасет для
event-classification: те же события на ru/eng новостях получают похожий sentiment.

Live-fetch отсутствует; читаем кэш с https://nbsdc.cn/ если есть.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


# Mапа event-типа в sentiment-сдвиг
EVENT_SENTIMENT = {
    "earnings_beat":   0.5,
    "earnings_miss":  -0.5,
    "dividend_increase": 0.4,
    "dividend_cut":   -0.4,
    "share_buyback":   0.3,
    "lawsuit":        -0.3,
    "merger":          0.2,
    "acquisition":     0.2,
    "guidance_up":     0.4,
    "guidance_down":  -0.4,
    "default":         0.0,
}


class NBSDCSource(NewsSource):
    DATA_PATH = Path("data/datasets/nbsdc/recent.jsonl")

    def fetch(self, since: datetime) -> List[NewsItem]:
        if not self.DATA_PATH.exists():
            return []
        import json
        items: List[NewsItem] = []
        try:
            with open(self.DATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    try:
                        pub = datetime.fromisoformat(row["timestamp"])
                    except Exception:
                        continue
                    if pub < since:
                        continue
                    event_type = row.get("event_type", "default")
                    base = EVENT_SENTIMENT.get(event_type, 0.0)
                    popularity = float(row.get("emotional_popularity", 0.5))   # [0, 1]
                    sentiment = base * (0.5 + popularity * 0.5)
                    items.append(self.parse_item(
                        title=row.get("headline", ""),
                        summary=f"{event_type} · {row.get('industry', '')}",
                        url=row.get("url"),
                        published_at=pub,
                        explicit_sentiment=sentiment,
                        explicit_tickers=row.get("tickers"),
                    ))
        except Exception as e:
            logger.warning("NBSDC read failed: %s", e)
        return items
