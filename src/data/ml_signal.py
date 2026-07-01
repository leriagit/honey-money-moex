"""
ML-сигнал: интерфейс провайдера + baseline + LightGBM-stub + Macro overlay.

Финальную модель (LightGBM/CatBoost на Alpha158, Chronos-Bolt) подключаем
через тот же интерфейс. MacroContextMLProvider оборачивает любой базовый
провайдер и применяет explainable logit-shift на основе макро-приоров
из config/macro_priors.yaml — даёт гибридную модель LightGBM × Macro.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional, Callable
import math

import numpy as np

from ..schemas import Candle, MLSignal, NewsSignal
from .indicators import rsi, macd, sma
from .macro_context import (
    MacroContext,
    LogitDecomposition,
    prob_to_logit,
    logit_to_prob,
)


class MLProvider(ABC):
    """Контракт ML-провайдера.

    Параметр `ticker` опционален для обратной совместимости — старые
    провайдеры (Baseline, LightGBM-stub) его игнорируют. MacroContext
    использует его обязательно. Если ticker не передан и его нельзя
    извлечь из news.ticker — overlay даёт passthrough.
    """

    @abstractmethod
    def predict(
        self,
        candles: List[Candle],
        news: Optional[NewsSignal] = None,
        horizon: str = "1h",
        ticker: Optional[str] = None,
    ) -> MLSignal:
        ...


# ────────────────────── Baseline-stub ────────────────────────────


class BaselineMLProvider(MLProvider):
    """
    Лёгкий baseline. Считает P(up) как логит от комбинации:
    - наклона цены за последние N баров
    - RSI (откат от экстремумов в свою сторону)
    - MACD-гистограммы
    - news.score (если есть)

    Это НЕ финальная модель. Цель — дать SignalEngine что-то осмысленное,
    пока обучаемся на Qlib/Alpha158.
    """

    def __init__(
        self,
        lookback: int = 30,
        slope_weight: float = 4.0,
        rsi_weight: float = 0.04,
        macd_weight: float = 8.0,
        news_weight: float = 1.5,
        version: str = "baseline-v1",
    ) -> None:
        self.lookback = lookback
        self.slope_weight = slope_weight
        self.rsi_weight = rsi_weight
        self.macd_weight = macd_weight
        self.news_weight = news_weight
        self.version = version

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x > 30:
            return 1.0
        if x < -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def predict(
        self,
        candles: List[Candle],
        news: Optional[NewsSignal] = None,
        horizon: str = "1h",
        ticker: Optional[str] = None,
    ) -> MLSignal:
        if not candles:
            return MLSignal(prob_up=0.5, model_version=self.version, horizon=horizon)

        closes = np.array([c.close for c in candles], dtype=float)

        # Логит-компонент 1: нормированный наклон последнего отрезка
        tail = closes[-self.lookback:] if len(closes) >= self.lookback else closes
        if len(tail) >= 2 and tail[0] > 0:
            slope = (tail[-1] - tail[0]) / tail[0]   # доходность за окно
        else:
            slope = 0.0
        logit = self.slope_weight * slope

        # Логит-компонент 2: RSI (значение 50 нейтрально)
        if len(closes) >= 15:
            r = rsi(closes, 14)[-1]
            if not math.isnan(r):
                logit += self.rsi_weight * (r - 50.0)

        # Логит-компонент 3: MACD hist (нормирован по цене)
        if len(closes) >= 35:
            macd_line, sig_line, _ = macd(closes)
            if not math.isnan(macd_line[-1]) and closes[-1] > 0:
                hist = (macd_line[-1] - sig_line[-1]) / closes[-1]
                logit += self.macd_weight * hist

        # Логит-компонент 4: news score (если confidence достаточен)
        if news is not None and news.confidence >= 0.2:
            logit += self.news_weight * news.score * news.confidence

        prob_up = self._sigmoid(logit)

        # Ожидаемая доходность — наивная оценка через std
        if len(closes) >= 5:
            rets = np.diff(closes) / closes[:-1]
            vol = float(rets.std()) if len(rets) > 1 else 0.01
        else:
            vol = 0.01
        expected_return = (prob_up - 0.5) * 2 * vol
        return MLSignal(
            prob_up=float(prob_up),
            expected_return=expected_return,
            expected_vol=vol,
            horizon=horizon,
            model_version=self.version,
        )


# ────────────────────── Заглушка под боевую модель ───────────────


class LightGBMMLProvider(MLProvider):
    """
    Боевой LightGBM-провайдер.

    Загружает .txt модель, обученную через scripts/train_lightgbm.py,
    и считает фичи онлайн через src.data.features.featurize_single_window.

    Если модель или метаданные отсутствуют — graceful fallback на
    BaselineMLProvider, с понятным model_version для аудита.

    macro_snapshot можно передать через set_macro_snapshot() — это значения
    Brent/USD-RUB/CBR-rate/gold/IMOEX на текущий момент. Они должны
    совпадать с теми, что использовались при обучении (см. metrics JSON).
    """

    def __init__(
        self,
        model_path: str = "models/lightgbm_v1.txt",
        metrics_path: Optional[str] = None,
        macro_snapshot: Optional[dict] = None,
    ) -> None:
        self.model_path = model_path
        self.metrics_path = metrics_path or model_path.replace(".txt", "_metrics.json")
        self.macro_snapshot = macro_snapshot or {}
        self._model = None
        self._feature_cols: Optional[List[str]] = None
        self._cat_features: List[str] = []
        self._fallback = BaselineMLProvider()
        self._load_model()

    def _load_model(self) -> None:
        try:
            import lightgbm as lgb  # type: ignore
            import os, json
            if not os.path.exists(self.model_path):
                return
            self._model = lgb.Booster(model_file=self.model_path)
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path) as f:
                    meta = json.load(f).get("meta", {})
                self._feature_cols = meta.get("feature_columns")
                self._cat_features = meta.get("categorical_features", [])
        except Exception:
            self._model = None

    def set_macro_snapshot(self, snapshot: dict) -> None:
        """Обновить текущий снимок макро (вызывается раз в день / при обновлении)."""
        self.macro_snapshot = dict(snapshot)

    # ─────── helpers ───────

    @staticmethod
    def _candles_to_df(candles: List[Candle]):
        import pandas as pd
        return pd.DataFrame([{
            "ts": c.ts,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in candles])

    def predict(
        self,
        candles: List[Candle],
        news: Optional[NewsSignal] = None,
        horizon: str = "1h",
        ticker: Optional[str] = None,
    ) -> MLSignal:
        # Fallback если модель не загрузилась
        if self._model is None:
            sig = self._fallback.predict(candles, news, horizon, ticker=ticker)
            return sig.model_copy(update={"model_version": "lightgbm-fallback->baseline-v1"})

        # Определяем ticker: явный аргумент или news.ticker
        tk = ticker
        if tk is None and news is not None:
            tk = getattr(news, "ticker", None)
        if tk is None:
            tk = "UNKN"

        # Нужно ≥60 баров для всех роллингов; иначе fallback
        if not candles or len(candles) < 60:
            sig = self._fallback.predict(candles, news, horizon, ticker=tk)
            return sig.model_copy(update={"model_version": "lightgbm-fallback->baseline-v1[short]"})

        # Featurize онлайн
        try:
            from .features import featurize_single_window, FeatureConfig
            df = self._candles_to_df(candles)
            feats = featurize_single_window(
                df,
                macro_snapshot=self.macro_snapshot,
                ticker=tk,
                config=FeatureConfig(dropna_target_rows=False),
            )
            if feats is None:
                raise ValueError("featurize returned None")

            # отбираем колонки в том же порядке, что при обучении
            if self._feature_cols is None:
                # без метаданных — берём всё что есть, в алфавитном порядке
                X = feats.sort_index()
            else:
                # отсутствующие колонки заполняем 0 (например, gold_log если macro не задан)
                X = []
                for col in self._feature_cols:
                    X.append(float(feats.get(col, 0.0)))
                import numpy as np
                X = np.array(X).reshape(1, -1)

            prob_up_raw = float(self._model.predict(X)[0])
            # клиппинг для безопасности
            prob_up = max(0.001, min(0.999, prob_up_raw))

        except Exception as e:
            sig = self._fallback.predict(candles, news, horizon, ticker=tk)
            return sig.model_copy(
                update={"model_version": f"lightgbm-fallback->baseline-v1[err:{type(e).__name__}]"}
            )

        # Оценка expected_return через дельту prob_up и эмпирическую vol
        closes = np.array([c.close for c in candles], dtype=float)
        rets = np.diff(closes) / closes[:-1] if len(closes) >= 2 else np.array([0.01])
        vol = float(rets.std()) if len(rets) > 1 else 0.01
        expected_return = (prob_up - 0.5) * 2.0 * vol

        return MLSignal(
            prob_up=prob_up,
            expected_return=expected_return,
            expected_vol=vol,
            horizon=horizon,
            model_version=f"lightgbm-v1|p={prob_up:.3f}|t={tk}",
        )


# ────────────────────── Macro overlay (Phase 1) ──────────────────


class MacroContextMLProvider(MLProvider):
    """
    Macro-aware overlay поверх любого базового MLProvider.

    Алгоритм:
      1. base_sig = base.predict(candles, news, horizon, ticker=ticker)
      2. ticker определяется по аргументу либо из news.ticker
      3. macro.compute_logit_shift(ticker, now) → LogitDecomposition
      4. Если ticker в HOLD-окне бинарного события — форсируем prob_up = 0.5
      5. Иначе: новый logit = logit(base_prob_up) + decomp.total
                новый prob_up = sigmoid(новый logit)
      6. expected_return пересчитывается из дельты prob_up и base_vol
      7. model_version содержит компактное разложение для аудит-лога

    Это explainable overlay: каждый сдвиг логита прозрачен,
    эксперты на защите видят, какие именно макро-факторы сработали.

    Использование:
      base = LightGBMMLProvider("models/lightgbm.txt")  # или BaselineMLProvider()
      macro = MacroContext("config/macro_priors.yaml")
      provider = MacroContextMLProvider(base=base, macro=macro)
      sig = provider.predict(candles, news, horizon="1h", ticker="SBER")
    """

    def __init__(
        self,
        base: MLProvider,
        macro: MacroContext,
        now: Optional[Callable[[], datetime]] = None,
        version: str = "macro-overlay-v1",
    ) -> None:
        self.base = base
        self.macro = macro
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.version = version

    # ─────── helpers ───────

    def _resolve_ticker(
        self,
        ticker: Optional[str],
        news: Optional[NewsSignal],
    ) -> Optional[str]:
        if ticker:
            return ticker
        if news is not None and getattr(news, "ticker", None):
            return news.ticker
        return None

    # ─────── main ───────

    def predict(
        self,
        candles: List[Candle],
        news: Optional[NewsSignal] = None,
        horizon: str = "1h",
        ticker: Optional[str] = None,
    ) -> MLSignal:
        base_sig = self.base.predict(candles, news, horizon, ticker=ticker)
        tk = self._resolve_ticker(ticker, news)

        # passthrough в трёх случаях:
        #   1) конфиг не загружен
        #   2) ticker не определён
        #   3) тикер неизвестен в конфиге, и passthrough разрешён
        if not self.macro.loaded:
            return base_sig.model_copy(
                update={"model_version": f"{self.version}->{base_sig.model_version} [no-config]"}
            )
        if tk is None:
            return base_sig.model_copy(
                update={"model_version": f"{self.version}->{base_sig.model_version} [no-ticker]"}
            )
        if tk not in self.macro.tickers:
            if self.macro.overlay_cfg.default_unknown_ticker_passthrough:
                return base_sig.model_copy(
                    update={"model_version": f"{self.version}->{base_sig.model_version} [unk:{tk}]"}
                )
            # иначе — продолжаем, но decomposition будет пустой

        now = self._now()
        # Гарантируем offset-aware UTC
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        decomp = self.macro.compute_logit_shift(tk, now)

        # Event gating: если в HOLD-окне бинарного события — форсируем prob_up = 0.5
        # SignalEngine увидит neutral ML и не даст ни BUY, ни SELL по ML-компоненту.
        if (
            self.macro.overlay_cfg.enable_event_gating
            and decomp.holding_event_id is not None
        ):
            return MLSignal(
                prob_up=0.5,
                expected_return=0.0,
                expected_vol=base_sig.expected_vol,
                horizon=horizon,
                model_version=(
                    f"{self.version}|base={base_sig.model_version}"
                    f"|HOLD:{decomp.holding_event_id}|{decomp.to_rationale()}"
                ),
            )

        # Логит-сложение
        z_base = prob_to_logit(base_sig.prob_up)
        z_new = z_base + decomp.total
        prob_up_new = logit_to_prob(z_new)

        # Пересчёт expected_return через дельту prob_up и vol
        vol = base_sig.expected_vol if base_sig.expected_vol is not None else 0.01
        expected_return_new = (prob_up_new - 0.5) * 2.0 * vol

        return MLSignal(
            prob_up=float(prob_up_new),
            expected_return=float(expected_return_new),
            expected_vol=vol,
            horizon=horizon,
            model_version=(
                f"{self.version}|base={base_sig.model_version}"
                f"|p_base={base_sig.prob_up:.3f}|p_new={prob_up_new:.3f}"
                f"|{decomp.to_rationale()}"
            ),
        )
