"""ФИНАМ Alert — старейший брокер РФ, торговые сигналы. Вес 0.85. Через Telegram."""
from ._telegram_common import TelegramChannelSource


class FinamSource(TelegramChannelSource):
    CHANNEL = "finamalert"
