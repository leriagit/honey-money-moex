"""
Синтетический news_signal на основе macro_priors.yaml.

Используется в:
  - backtest: на прошлом периоде у нас нет live новостей, но есть карта
    macro_state. Транслируем её в NewsSignal с consistent score/confidence.
  - production fallback: если NewsAggregator упал (rate limit, network),
    можно использовать static synthetic как backup.

Логика трансляции:
  - суммируем все active macro flags с весами из tickers.<TICKER>.factors
  - normalize в [-1, 1] для score
  - confidence = функция от количества активных факторов
  - ru_defense_flag = True если есть >= 2 негативных регулярных фактора
"""
from __future__ import annotations

import math
from typing import Optional

from ...schemas import NewsSignal
from ..macro_context import MacroContext


def build_news_signal_from_priors(
    ticker: str,
    macro: MacroContext,
    base_confidence: float = 0.35,
    score_scale: float = 0.5,
) -> NewsSignal:
    """
    Создаёт NewsSignal на основе текущих macro flags и priors для тикера.

    Это НЕ замена реальному NewsAggregator (у которого 37 источников
    с разными весами), а упрощённый source-free сигнал.

    Используется:
      - В backtest как proxy для news (раньше всегда был score=0)
      - В проде как fallback если NewsAggregator недоступен

    Args:
        ticker: тикер, например "SBER"
        macro: загруженный MacroContext
        base_confidence: базовая confidence (0..1), если есть хотя бы 1 active flag
        score_scale: scale factor для перевода logit-shift'а в score [-1, 1]
    """
    if not macro.loaded or ticker not in macro.tickers:
        return NewsSignal(ticker=ticker, score=0.0, confidence=0.0)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    decomp = macro.compute_logit_shift(ticker, now)

    # score = scaled sum of contributions
    # Используем tanh для clipping в [-1, 1]
    total = decomp.total
    score = float(math.tanh(total * score_scale))

    # confidence = функция от количества активных факторов
    n_factors = (
        len([v for v in decomp.flag_contributions.values() if abs(v) > 1e-6])
        + len([v for v in decomp.event_contributions.values() if abs(v) > 1e-6])
    )
    if n_factors == 0:
        confidence = 0.0
    else:
        # 1 фактор → 0.35, 5 факторов → 0.75, 10+ → cap 0.9
        confidence = min(0.9, base_confidence + 0.1 * (n_factors - 1))

    # ru_defense_flag — если score < -0.3 и достаточно факторов
    ru_defense_flag = bool(score < -0.3 and n_factors >= 2)

    return NewsSignal(
        ticker=ticker,
        score=score,
        confidence=confidence,
        n_items=n_factors,
        ru_defense_flag=ru_defense_flag,
    )
