"""Альфа-Инвестиции — официальный канал брокера. Лицензия ЦБ, верифицировано. Вес 0.85."""
from ._telegram_common import TelegramChannelSource


class AlfaInvestSource(TelegramChannelSource):
    CHANNEL = "alfa_investments"
