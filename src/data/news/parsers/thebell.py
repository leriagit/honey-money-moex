"""
The Bell — деловое издание с расследованиями. Раскрыли RDV-памп.
Высочайшее доверие. Вес 0.90.
"""
from ._telegram_common import TelegramChannelSource


class TheBellSource(TelegramChannelSource):
    CHANNEL = "thebell_io"
