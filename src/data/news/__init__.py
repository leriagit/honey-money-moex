"""
News-модуль: 19 источников из ТЗ + RU-фильтр + Trump-priority + агрегатор.

Главные точки входа:
- registry.load_sources()                — загрузить конфиг с весами
- aggregator.NewsAggregator.aggregate()  — взвешенный sentiment по тикеру
"""
from .base import NewsSource, NewsFetchError
from .registry import load_sources, get_source_weight
from .aggregator import NewsAggregator

__all__ = [
    "NewsSource",
    "NewsFetchError",
    "load_sources",
    "get_source_weight",
    "NewsAggregator",
]
