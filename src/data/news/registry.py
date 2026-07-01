"""
Реестр источников. Грузит config/news_sources.yaml и инстанцирует парсеры.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .base import NewsSource, SourceConfig


# Импорт всех парсеров — ленивая инициализация в _build_source()
_PARSER_REGISTRY: Dict[str, str] = {
    # Оригинальные 19 источников
    "finmarba":      "parsers.finmarba:FinMarBaSource",
    "stockemotions": "parsers.stockemotions:StockEmotionsSource",
    "nbsdc":         "parsers.nbsdc:NBSDCSource",
    "marketpsych":   "parsers.marketpsych:MarketPsychSource",
    "bloomberg":     "parsers.bloomberg:BloombergSource",
    "reuters":       "parsers.reuters:ReutersSource",
    "investing":     "parsers.investing:InvestingSource",
    "permutable":    "parsers.permutable:PermutableSource",
    "ravenpack":     "parsers.ravenpack:RavenPackSource",
    "assymetrix":    "parsers.assymetrix:AssymetrixSource",
    "quiver":        "parsers.quiver:QuiverSource",
    "ailantroquant": "parsers.ailantroquant:AilantroQuantSource",
    "alphavantage":  "parsers.alphavantage:AlphaVantageSource",
    "nyt":           "parsers.nyt:NYTSource",
    "wsj":           "parsers.wsj:WSJSource",
    "ft":            "parsers.ft:FTSource",
    "foxnews":       "parsers.foxnews:FoxNewsSource",
    "iea":           "parsers.iea:IEASource",
    "wapo":          "parsers.wapo:WaPoSource",
    # 10 новых ТГК
    "markettwits":   "parsers.markettwits:MarketTwitsSource",
    "dohod":         "parsers.dohod:DohodSource",
    "thebell":       "parsers.thebell:TheBellSource",
    "smartlab":      "parsers.smartlab:SmartlabSource",
    "investera":     "parsers.investera:InvestEraSource",
    "kotfin":        "parsers.kotfin:KotFinSource",
    "divonline":     "parsers.divonline:DivOnlineSource",
    "kbecon":        "parsers.kbecon:KBEconSource",
    "alfa_invest":   "parsers.alfainvest:AlfaInvestSource",
    "prostoecon":    "parsers.prostoecon:ProstoEconSource",
    # 8 ключевых российских
    "cbr":           "parsers.cbr:CBRSource",
    "kremlin":       "parsers.kremlin:KremlinSource",
    "moex_news":     "parsers.moex_news:MoexNewsSource",
    "rbc":           "parsers.rbc:RBCSource",
    "sbercib":       "parsers.sbercib:SberCIBSource",
    "tbank":         "parsers.tbank:TBankSource",
    "finam":         "parsers.finam:FinamSource",
    "bcs":           "parsers.bcs:BCSExpressSource",
}


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "news_sources.yaml"


def _load_yaml(path: Optional[Path] = None) -> dict:
    p = path or _default_config_path()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_source_weight(source_id: str, path: Optional[Path] = None) -> float:
    """Дёшевая read-only функция: вернуть вес одного источника."""
    cfg = _load_yaml(path)
    for s in cfg.get("sources", []):
        if s["id"] == source_id:
            return float(s.get("weight", 0.0))
    return 0.0


def _build_source(spec: dict) -> Optional[NewsSource]:
    sid = spec["id"]
    target = _PARSER_REGISTRY.get(sid)
    if not target:
        return None

    module_path, class_name = target.rsplit(":", 1)
    module_full = f"{__package__}.{module_path}"
    try:
        import importlib
        module = importlib.import_module(module_full)
        cls = getattr(module, class_name)
    except Exception:
        # Если конкретный парсер сломался — пропускаем, остальные продолжают работать
        return None

    cfg = SourceConfig(
        id=sid,
        name=spec.get("name", sid),
        weight=float(spec.get("weight", 0.0)),
        requires_key=bool(spec.get("requires_key", False)),
        env_key=spec.get("env_key"),
        rss_url=spec.get("rss_url"),
        ru_specialized=bool(spec.get("ru_specialized", False)),
        trump_priority=bool(spec.get("trump_priority", False)),
        trust_tier=str(spec.get("trust_tier", "verified")),
        regulator_triggers=spec.get("regulator_triggers"),
        extra=spec,
    )
    return cls(cfg)


def load_sources(path: Optional[Path] = None, include_keyless_only: bool = False) -> List[NewsSource]:
    """
    Возвращает список инстанцированных источников.

    include_keyless_only=True пропускает источники, требующие API-ключ,
    если ключ не задан в env. Удобно для smoke-теста и хакатона.
    """
    cfg = _load_yaml(path)
    sources: List[NewsSource] = []
    for spec in cfg.get("sources", []):
        if include_keyless_only and spec.get("requires_key"):
            env_key = spec.get("env_key")
            if not env_key or not os.environ.get(env_key):
                continue
        src = _build_source(spec)
        if src is not None:
            sources.append(src)
    return sources


def load_aggregator_config(path: Optional[Path] = None) -> dict:
    cfg = _load_yaml(path)
    return cfg.get("aggregator", {})


def load_ru_counter_sources(path: Optional[Path] = None) -> List[str]:
    cfg = _load_yaml(path)
    return list(cfg.get("ru_counter_sources", []))
