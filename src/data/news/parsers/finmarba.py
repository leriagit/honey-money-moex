"""
FinMarBa — вес 0.30. Датасет разметки новостей по реальной рыночной реакции.
arxiv: 2507.22932

Используется не для live-стрима новостей, а как калибровочный датасет:
из него мы извлекаем "семантические паттерны" — какие новости реально
двигают рынок vs какие нет. На рантайме отдаёт recent samples
(out-of-sample проверка).

В live-режиме fetch() возвращает пустой список — данные подключаем offline
к training-pipeline (см. scripts/train_ml.py, заглушка).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

from ....schemas import NewsItem
from ..base import NewsSource

logger = logging.getLogger(__name__)


class FinMarBaSource(NewsSource):
    """
    Если в data/datasets/finmarba/recent.jsonl лежит свежий слайс датасета —
    отдаём его как новости. Иначе пусто.
    """

    DATA_PATH = Path("data/datasets/finmarba/recent.jsonl")

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
                    items.append(self.parse_item(
                        title=row.get("headline", ""),
                        summary=row.get("body", ""),
                        url=row.get("url"),
                        published_at=pub,
                        explicit_sentiment=float(row.get("market_reaction", 0.0)),
                        raw_score=row.get("confidence"),
                    ))
        except Exception as e:
            logger.warning("FinMarBa read failed: %s", e)
        return items
