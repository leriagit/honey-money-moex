"""
Кот.Финанс — анонимный канал с инвест-идеями.
Возможна аффилированность через рекламу → понижаем доверие. Вес 0.40.
Trust tier: anonymous (см. логику в news_aggregator: при противоречии
с verified источниками сигнал Кот.Финанс дисконтируется).
"""
from ._telegram_common import TelegramChannelSource


class KotFinSource(TelegramChannelSource):
    CHANNEL = "KotFin"
