"""Простая экономика (Н.Мячин) — образовательный контент, не торговые сигналы. Вес 0.60."""
from ._telegram_common import TelegramChannelSource


class ProstoEconSource(TelegramChannelSource):
    CHANNEL = "prostoecon"
