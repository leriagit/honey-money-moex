"""КБ.Экономика — макроэкономика и мировые новости. Не торговые сигналы. Вес 0.70."""
from ._telegram_common import TelegramChannelSource


class KBEconSource(TelegramChannelSource):
    CHANNEL = "kbecon"
