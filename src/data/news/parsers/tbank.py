"""
Т-Банк Инвестиции / Тинькофф — крупнейшая розничная платформа.
Вес 0.80 (соцсеть-Пульс может быть шумной, понижаем).
Через Telegram.
"""
from ._telegram_common import TelegramChannelSource


class TBankSource(TelegramChannelSource):
    CHANNEL = "tinkoff_invest"
