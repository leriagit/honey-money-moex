"""SberCIB / Сбербанк — крупнейший брокер РФ. Вес 0.90. Через Telegram."""
from ._telegram_common import TelegramChannelSource


class SberCIBSource(TelegramChannelSource):
    CHANNEL = "sberinvestments"
