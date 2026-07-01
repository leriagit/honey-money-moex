"""bench_latency — воспроизводимый замер латентности ML-инференса на CPU.

Показывает production-метрику: за сколько модель переоценивает весь портфель.
Запуск: python -m scripts.bench_latency
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, timezone

from src.schemas import Candle
from src.data.ml_signal import LightGBMMLProvider

N_TICKERS = 20
N_ITERS = 200


def _synth_candles(n: int = 70) -> list:
    random.seed(1)
    px, base, out = 250.0, datetime(2026, 5, 1, tzinfo=timezone.utc), []
    for i in range(n):
        px *= (1 + random.uniform(-0.01, 0.011))
        out.append(Candle(ts=base + timedelta(hours=i), open=px, high=px * 1.004,
                          low=px * 0.996, close=px, volume=random.randint(10**5, 5 * 10**5)))
    return out


def main(model_path: str = "models/lightgbm_v1.txt") -> dict:
    candles = _synth_candles()
    t0 = time.perf_counter()
    provider = LightGBMMLProvider(model_path=model_path)
    load_ms = (time.perf_counter() - t0) * 1000
    if provider._model is None:
        print("ВНИМАНИЕ: модель не загрузилась — измеряется fallback.")

    for _ in range(5):  # прогрев
        provider.predict(candles, ticker="SBER")

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        provider.predict(candles, ticker="SBER")
    per_ticker_ms = (time.perf_counter() - t0) / N_ITERS * 1000

    res = {
        "load_ms": round(load_ms, 1),
        "per_ticker_ms": round(per_ticker_ms, 2),
        "full_cycle_ms": round(per_ticker_ms * N_TICKERS, 1),
    }
    print(f"load model:           {res['load_ms']:.0f} ms (однократно при старте)")
    print(f"инференс / тикер:      {res['per_ticker_ms']:.2f} ms")
    print(f"цикл {N_TICKERS} тикеров:       {res['full_cycle_ms']:.1f} ms  (CPU, без GPU)")
    return res


if __name__ == "__main__":
    main()
