"""
Базовый класс для всех LLM-агентов в multi_agent_design.

Контракт:
  - Каждый агент реализует analyze(context) -> AgentOutput
  - При недоступности LLM (нет ключа, rate limit, network) — graceful fallback
    на rule-based реализацию через fallback() метод
  - Все агенты возвращают структурированный JSON с reasoning
  - Никогда не падают — возвращают AgentOutput с success=False

Использование:
    class MyAgent(LLMAgent):
        def get_model(self): return "qwen-2.5-14b"
        def build_prompt(self, ctx): return [...]
        def fallback(self, ctx): return {...}  # rule-based

Архитектурный выбор: НЕ используем LangGraph (overkill).
Простая sequential orchestration на Python — её легче дебажить,
не требует внешних зависимостей сверх того что у нас есть.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .polza_client import PolzaClient, PolzaResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Результат работы одного агента."""
    agent_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    used_fallback: bool = False
    model: str = ""
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LLMAgent(ABC):
    """Базовый класс. Подклассы переопределяют build_prompt + parse + fallback."""

    name: str = "unnamed-agent"

    def __init__(
        self,
        client: Optional[PolzaClient] = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        enable: bool = True,  # можно отключить агента полностью
    ) -> None:
        self.client = client or PolzaClient()
        self.json_mode = json_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable = enable

    # ─────── абстрактные методы ───────

    @abstractmethod
    def get_model(self) -> str:
        """Какую модель использовать (например 'qwen-2.5-14b')."""
        ...

    @abstractmethod
    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Промпт в формате [{role: ..., content: ...}, ...]"""
        ...

    @abstractmethod
    def fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based фоллбек при недоступности LLM.
        ОБЯЗАТЕЛЬНО реализовать — это safety net для прода (этап 2,
        где нельзя удалённо вмешиваться).
        """
        ...

    def parse_response(self, response: PolzaResponse) -> Dict[str, Any]:
        """Парсит ответ модели. Дефолтная реализация — берёт parsed_json."""
        if response.parsed_json:
            return response.parsed_json
        return {"raw": response.content}

    # ─────── главный метод ───────

    def analyze(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Главный метод. Возвращает AgentOutput.

        Алгоритм:
          1. Если agent отключён (enable=False) → сразу fallback
          2. Если LLM не настроен → fallback
          3. Зовём LLM. Если упал → fallback
          4. Парсим. Если JSON некорректный → fallback
        """
        # 1) Disabled — сразу fallback
        if not self.enable:
            return self._wrap_fallback(context, reason="agent disabled")

        # 2) LLM не настроен
        if not self.client.is_configured:
            return self._wrap_fallback(context, reason="POLZA_API_KEY not configured")

        # 3) LLM call
        try:
            messages = self.build_prompt(context)
        except Exception as e:
            logger.exception("%s: build_prompt failed", self.name)
            return self._wrap_fallback(context, reason=f"build_prompt: {e}")

        response = self.client.chat(
            model=self.get_model(),
            messages=messages,
            json_mode=self.json_mode,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.success:
            return self._wrap_fallback(context, reason=f"LLM failed: {response.error}")

        # 4) Парсим
        try:
            data = self.parse_response(response)
        except Exception as e:
            logger.exception("%s: parse_response failed", self.name)
            return self._wrap_fallback(context, reason=f"parse: {e}")

        if not data:
            return self._wrap_fallback(context, reason="empty parsed_json")

        return AgentOutput(
            agent_name=self.name,
            success=True,
            data=data,
            rationale=data.get("rationale") or data.get("narrative", ""),
            used_fallback=False,
            model=response.model,
            elapsed_sec=response.elapsed_sec,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def _wrap_fallback(self, context: Dict[str, Any], reason: str) -> AgentOutput:
        try:
            data = self.fallback(context)
        except Exception as e:
            logger.exception("%s: fallback failed", self.name)
            data = {}
        return AgentOutput(
            agent_name=self.name,
            success=bool(data),
            data=data,
            rationale=data.get("rationale", f"fallback (rule-based): {reason}"),
            used_fallback=True,
            error=reason,
        )

    # ─────────── Async версия для параллельного выполнения ───────────

    async def async_analyze(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Async версия analyze(). Логически идентична sync версии, но
        использует client.async_chat() для возможности параллельного
        выполнения через asyncio.gather().
        """
        if not self.enable:
            return self._wrap_fallback(context, reason="agent disabled")

        if not self.client.is_configured:
            return self._wrap_fallback(context, reason="POLZA_API_KEY not configured")

        try:
            messages = self.build_prompt(context)
        except Exception as e:
            logger.exception("%s: build_prompt failed", self.name)
            return self._wrap_fallback(context, reason=f"build_prompt: {e}")

        response = await self.client.async_chat(
            model=self.get_model(),
            messages=messages,
            json_mode=self.json_mode,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.success:
            return self._wrap_fallback(context, reason=f"LLM failed: {response.error}")

        try:
            data = self.parse_response(response)
        except Exception as e:
            logger.exception("%s: parse_response failed", self.name)
            return self._wrap_fallback(context, reason=f"parse: {e}")

        if not data:
            return self._wrap_fallback(context, reason="empty parsed_json")

        return AgentOutput(
            agent_name=self.name,
            success=True,
            data=data,
            rationale=data.get("rationale") or data.get("narrative", ""),
            used_fallback=False,
            model=response.model,
            elapsed_sec=response.elapsed_sec,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )
