"""
StockEmotions — вес 0.50. Датасет комментариев инвесторов с разметкой эмоций.
arxiv: 2301.09279

Аналогично FinMarBa: грузим из локального jsonl, если положен — иначе пусто.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


# Маппинг эмоций StockEmotions в sentiment-шкалу [-1, 1].
EMOTION_TO_SENTIMENT = {
    "happy":     0.7,
    "optimism":  0.6,
    "excited":   0.5,
    "neutral":   0.0,
    "anxious":  -0.3,
    "ambiguous": 0.0,
    "panic":    -0.8,
    "depressed":-0.7,
    "surprise":  0.0,
    "anger":    -0.6,
    "disgust":  -0.5,
    "sad":      -0.6,
}


class StockEmotionsSource(NewsSource):
    DATA_PATH = Path("data/datasets/stockemotions/recent.jsonl")

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
                    emo = row.get("emotion", "neutral").lower()
                    sentiment = EMOTION_TO_SENTIMENT.get(emo, 0.0)
                    items.append(self.parse_item(
                        title=row.get("text", "")[:120],
                        summary=row.get("text", ""),
                        url=row.get("url"),
                        published_at=pub,
                        explicit_sentiment=sentiment,
                        explicit_tickers=row.get("tickers"),
                    ))
        except Exception as e:
            logger.warning("StockEmotions read failed: %s", e)
        return items
