"""
NewsAggregator: собирает NewsItem-ы из всех источников и
агрегирует их в NewsSignal по каждому тикеру.

Правила (из ТЗ):
1. Каждый источник имеет вес w_i ∈ [0..1].
2. RU-релевантные новости получают boost = ru_relevance_boost (по умолчанию 1.5).
3. Новости с упоминанием Трампа/Маска/Вэнса получают boost = trump_boost (1.3).
4. Итоговый score = Σ(sentiment_i * effective_weight_i) / Σ(effective_weight_i).
5. ru_defense_flag: если РУ-источники (РБК/Финам/Интерфакс) дефают
   плохую новость → она скорее всего ложь → ШОРТ (флаг ставится отдельно,
   обработка решения в SignalEngine/Provider).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from ...schemas import NewsItem, NewsSignal
from .base import NewsSource
from .registry import (
    load_sources,
    load_aggregator_config,
    load_ru_counter_sources,
)

logger = logging.getLogger(__name__)


class NewsAggregator:
    def __init__(
        self,
        sources: Optional[List[NewsSource]] = None,
        ru_relevance_boost: Optional[float] = None,
        trump_boost: Optional[float] = None,
        window_hours: Optional[int] = None,
        min_confidence: Optional[float] = None,
        enable_ru_defense: Optional[bool] = None,
        ru_counter_ids: Optional[List[str]] = None,
        trust_multipliers: Optional[Dict[str, float]] = None,
    ) -> None:
        cfg = load_aggregator_config()
        self.sources = sources if sources is not None else load_sources(include_keyless_only=True)
        self.ru_relevance_boost = ru_relevance_boost if ru_relevance_boost is not None \
            else float(cfg.get("ru_relevance_boost", 1.5))
        self.trump_boost = trump_boost if trump_boost is not None \
            else float(cfg.get("trump_boost", 1.3))
        self.window_hours = window_hours if window_hours is not None \
            else int(cfg.get("window_hours", 24))
        self.min_confidence = min_confidence if min_confidence is not None \
            else float(cfg.get("min_confidence", 0.20))
        self.enable_ru_defense = enable_ru_defense if enable_ru_defense is not None \
            else bool(cfg.get("enable_ru_defense_signal", True))
        self.ru_counter_ids: Set[str] = set(ru_counter_ids or load_ru_counter_sources())
        # Trust-tier multipliers (применяется поверх weight)
        self.trust_multipliers: Dict[str, float] = dict(
            trust_multipliers if trust_multipliers is not None
            else cfg.get("trust_multipliers", {})
        )

    # ────────────────────── Получение данных ─────────────────────

    def collect(self, now: Optional[datetime] = None) -> List[NewsItem]:
        """Опрашивает все источники, собирает NewsItem за окно window_hours."""
        if now is None:
            now = datetime.now(timezone.utc)
        since = now - timedelta(hours=self.window_hours)
        all_items: List[NewsItem] = []
        for src in self.sources:
            try:
                items = src.fetch(since)
            except Exception as e:
                logger.warning("source %s crashed: %s", src.id, e)
                items = []
            if items:
                all_items.extend(items)
        # Дедупликация по (source_id, url или title)
        seen = set()
        dedup: List[NewsItem] = []
        for it in all_items:
            key = (it.source_id, it.url or it.title[:80])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        return dedup

    # ────────────────────── Агрегация ───────────────────────────

    def aggregate(
        self,
        items: List[NewsItem],
        ticker: str,
        external_news_items_by_ticker: Optional[Dict[str, List[NewsItem]]] = None,
    ) -> NewsSignal:
        """
        Считает NewsSignal по одному тикеру.

        Алгоритм:
        1. Фильтруем item-ы, где ticker есть в it.tickers — или ru_relevance > 0 (для общих макро-новостей)
        2. Эффективный вес = w_source * (1 + ru_boost*ru_relevance) * (1 + trump_boost*trump_mention)
        3. score = Σ(sentiment * eff_w) / Σ(eff_w)
        4. confidence = Σ(eff_w) / (n * max_w) — нормированная сила голоса.
        5. ru_defense_flag по контр-источникам.
        """
        # source_id -> вес из конфига × trust_multiplier(tier)
        source_weights: Dict[str, float] = {}
        source_tiers: Dict[str, str] = {}
        for s in self.sources:
            tier = getattr(s.cfg, "trust_tier", "verified")
            multiplier = self.trust_multipliers.get(tier, 1.0)
            source_weights[s.id] = s.cfg.weight * multiplier
            source_tiers[s.id] = tier
        max_source_weight = max(source_weights.values()) if source_weights else 1.0

        relevant: List[NewsItem] = []
        for it in items:
            if ticker in it.tickers:
                relevant.append(it)
            elif it.ru_relevance >= 0.4:
                # макро-новости с РУ-релевантностью трогают весь портфель
                relevant.append(it)

        if not relevant:
            return NewsSignal(ticker=ticker, score=0.0, confidence=0.0, n_items=0)

        sum_w = 0.0
        sum_sw = 0.0
        n_ru = 0
        n_trump = 0
        ru_counter_items: List[NewsItem] = []   # ВСЕ item-ы от RU-counter источников
        non_ru_counter_items: List[NewsItem] = []
        regulator_titles: List[str] = []
        regulator_event = False

        for it in relevant:
            w = source_weights.get(it.source_id, 0.5)
            eff = w
            if it.ru_relevance > 0:
                eff *= (1.0 + (self.ru_relevance_boost - 1.0) * it.ru_relevance)
                n_ru += 1
            if it.trump_mention:
                eff *= self.trump_boost
                n_trump += 1
            sum_w += eff
            sum_sw += eff * it.sentiment
            if it.source_id in self.ru_counter_ids:
                ru_counter_items.append(it)
            else:
                non_ru_counter_items.append(it)
            if it.regulator_event:
                regulator_event = True
                regulator_titles.append(f"[{it.source_id}] {it.title[:140]}")

        score = sum_sw / sum_w if sum_w > 0 else 0.0
        score = max(-1.0, min(1.0, score))

        # confidence = насколько собранный вес близок к максимально возможному
        # для N item-ов с весом max_source_weight
        max_possible = max_source_weight * len(relevant) * self.ru_relevance_boost
        confidence = sum_w / max_possible if max_possible > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))

        # ru_defense_flag из ТЗ:
        # "если RU-источники открыто дефают плохую новость – это скорее всего ложь"
        # Триггер: RU-counter источники в среднем НЕ-негативны (>= -0.2),
        # а остальные источники в среднем явно негативны (< -0.3).
        ru_defense_flag = False
        if (
            self.enable_ru_defense
            and ru_counter_items
            and non_ru_counter_items
        ):
            ru_avg = sum(i.sentiment for i in ru_counter_items) / len(ru_counter_items)
            others_avg = sum(i.sentiment for i in non_ru_counter_items) / len(non_ru_counter_items)
            if ru_avg >= -0.2 and others_avg < -0.3:
                ru_defense_flag = True

        # Сортируем top items по вкладу |sentiment * eff_w| для аудит-лога
        scored = []
        for it in relevant:
            w = source_weights.get(it.source_id, 0.5)
            eff = w * (1 + (self.ru_relevance_boost - 1) * it.ru_relevance)
            scored.append((abs(it.sentiment * eff), it))
        scored.sort(key=lambda x: -x[0])
        top = [it for _, it in scored[:5]]

        return NewsSignal(
            ticker=ticker,
            score=score,
            confidence=confidence,
            n_items=len(relevant),
            n_ru_relevant=n_ru,
            n_trump=n_trump,
            ru_defense_flag=ru_defense_flag,
            regulator_event=regulator_event,
            regulator_event_titles=regulator_titles[:5],
            top_items=top,
        )

    # ─────────────────── Удобный one-shot helper ────────────────

    def get_signal(self, ticker: str, now: Optional[datetime] = None) -> NewsSignal:
        items = self.collect(now)
        return self.aggregate(items, ticker)
