"""
Дивиденды Онлайн — агрегатор дивидендных новостей, без аналитики.
Sentiment всегда нейтральный (0.0) — канал просто констатирует факты.
Полезен как сигнал ex-date / dividend cut. Вес 0.50.
"""
from typing import Optional

from ._telegram_common import TelegramChannelSource


class DivOnlineSource(TelegramChannelSource):
    CHANNEL = "divonline"

    # Эвристика: "увеличил" / "повысил" → +, "снизил" / "отказался" → -
    DIVIDEND_POS = ["увеличил", "повысил", "рекордн", "выплачивает", "распределит"]
    DIVIDEND_NEG = ["отказ", "не платит", "снизил", "пропустил", "перенесён", "перенесен"]

    def sentiment_hint(self, text: str) -> Optional[float]:
        t = text.lower()
        if any(w in t for w in self.DIVIDEND_POS):
            return 0.4
        if any(w in t for w in self.DIVIDEND_NEG):
            return -0.5
        return 0.0
