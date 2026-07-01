"""MarketTwits — короткие посты, финансовые новости РФ и мира. Trust 5/5. Вес 0.85."""
from ._telegram_common import TelegramChannelSource


class MarketTwitsSource(TelegramChannelSource):
    CHANNEL = "markettwits"
