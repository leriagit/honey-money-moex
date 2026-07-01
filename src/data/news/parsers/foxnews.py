"""
Fox News — RSS. Вес 0.95, trump_priority=True.

Fox News часто первым публикует слова Трампа/Маска/Вэнса.
Логика приоритета Трампа реализована в base.parse_item():
если trump_priority=True И в новости есть упоминание Трампа,
ru_relevance поднимается до >=0.3 — это даст ему повышенный вес
в агрегаторе.
"""
from ._rss_common import RSSSource


class FoxNewsSource(RSSSource):
    pass
