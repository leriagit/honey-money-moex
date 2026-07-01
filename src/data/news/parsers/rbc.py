"""
РБК / RBC — флагман российских деловых новостей. Вес 0.95.
Также включается в ru_counter_ids для логики ru_defense_flag.
RSS публичный.
"""
from ._rss_common import RSSSource


class RBCSource(RSSSource):
    """Использует rss_url из конфига."""
    pass
