"""СМАРТЛАБ — комьюнити инвесторов, аналитика, форум. Вес 0.75."""
from ._telegram_common import TelegramChannelSource


class SmartlabSource(TelegramChannelSource):
    CHANNEL = "smartlabnews"
