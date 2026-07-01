"""BCS Express — быстрые новости и инвестидеи. Вес 0.85. Через Telegram."""
from ._telegram_common import TelegramChannelSource


class BCSExpressSource(TelegramChannelSource):
    CHANNEL = "bcs_express"
