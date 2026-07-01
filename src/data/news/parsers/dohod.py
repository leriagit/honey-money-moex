"""ДОХОДЪ — корпоративный канал УК. Дивидендный анализ, прогнозы. Вес 0.85."""
from ._telegram_common import TelegramChannelSource


class DohodSource(TelegramChannelSource):
    CHANNEL = "dohod"
