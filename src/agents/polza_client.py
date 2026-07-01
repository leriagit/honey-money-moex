"""
PolzaClient — клиент к Polza.ai (OpenAI-совместимый API).

Polza.ai даёт доступ к open-source моделям (DeepSeek, Qwen, Mistral)
с лицензией для коммерческого использования. Это требование ТЗ:
  > интеллектуальный агент на основе открытого генеративного ИИ

Бюджет: 6000₽ на каждый этап. Эконом-режим использует Qwen 2.5 14B для
лёгких задач, тяжёлый DeepSeek-V3 — только для финального reasoning'а.

API: OpenAI-compatible, базовый URL https://polza.ai/api/v1
Документация: https://polza.ai/docs

Поведение:
  - Никогда не бросает исключений — всегда возвращает PolzaResponse
  - При ошибках (rate limit, network, JSON parse) возвращает success=False
  - Поддерживает structured output через response_format={"type": "json_object"}
  - Логирует token usage для контроля бюджета
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..data._http import urlopen as _urlopen
import urllib.request

logger = logging.getLogger(__name__)


# Open-source модели со СВОБОДНОЙ КОММЕРЧЕСКОЙ лицензией (правило ТЗ).
# НЕЛЬЗЯ: gpt-4o, claude-*, gemini-*, grok-*,
# а также mistralai/mistral-large (= Mistral Large 2 2407, MRL=research only)
# Проверено через Polza API 27.05.2026 (GET /api/v1/models/{id}):
ALLOWED_MODELS = {
    "deepseek-chat":
        "DeepSeek-V3 (MIT) — мощный reasoning, резервная модель",
    "mistralai/mistral-large-2512":
        "Mistral Large 3 2512 (Apache 2.0) — основная модель LLM-агентов. "
        "Подтверждено Polza: 'released under the Apache 2.0 license'.",
}


@dataclass
class PolzaResponse:
    success: bool
    content: str = ""
    parsed_json: Optional[Dict[str, Any]] = None
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    error: Optional[str] = None
    elapsed_sec: float = 0.0


class PolzaClient:
    """
    OpenAI-compatible HTTP клиент к Polza.ai.

    Использование:
        client = PolzaClient(api_key=os.environ["POLZA_API_KEY"])
        resp = client.chat(
            model="qwen-2.5-14b",
            messages=[{"role": "user", "content": "..."}],
            json_mode=True,
        )
        if resp.success and resp.parsed_json:
            print(resp.parsed_json)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 180.0,  # DeepSeek на бесплатном Polza часто медленный
        max_retries: int = 2,
    ) -> None:
        # api_key=None → берём из env (типичное использование)
        # api_key="" → явно "без ключа" (для тестов и fallback-проверок)
        if api_key is None:
            self.api_key = os.environ.get("POLZA_API_KEY", "")
        else:
            self.api_key = api_key
        self.base_url = (
            base_url or os.environ.get("POLZA_BASE_URL", "https://polza.ai/api/v1")
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        # Учёт токенов на сессию — для контроля бюджета
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_calls = 0
        self.failed_calls = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> PolzaResponse:
        """
        Основной метод: посылает сообщения, возвращает ответ.
        НИКОГДА не бросает исключений — для использования в production без try/except.
        """
        if not self.is_configured:
            return PolzaResponse(
                success=False,
                error="POLZA_API_KEY не задан",
                model=model,
            )

        if model not in ALLOWED_MODELS:
            logger.warning("Polza: модель %s не в ALLOWED_MODELS — рискуем дисквалификацией", model)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Optional[str] = None
        t0 = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with _urlopen(req, timeout=self.timeout) as r:
                    raw = r.read().decode("utf-8")
                    data = json.loads(raw)

                # OpenAI format: choices[0].message.content
                content = ""
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    last_error = "malformed response"
                    continue

                usage = data.get("usage", {})
                tokens_in = int(usage.get("prompt_tokens", 0))
                tokens_out = int(usage.get("completion_tokens", 0))

                self.total_tokens_in += tokens_in
                self.total_tokens_out += tokens_out
                self.total_calls += 1

                parsed = None
                if json_mode:
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        # модель вернула не валидный JSON — пытаемся вытащить
                        parsed = self._extract_json(content)

                return PolzaResponse(
                    success=True,
                    content=content,
                    parsed_json=parsed,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                    elapsed_sec=time.time() - t0,
                )

            except urllib.error.HTTPError as e:
                last_error = f"HTTPError: HTTP Error {e.code}: {e.reason}"
                # 429/timeout — это ожидаемая ситуация degraded mode, debug-level
                level = logger.debug if e.code in (429, 408, 503, 504) else logger.warning
                level("Polza %s attempt %d/%d failed: %s",
                      model, attempt + 1, self.max_retries + 1, last_error)
                # 429 (rate limit) — длинный backoff
                if e.code == 429:
                    if attempt < self.max_retries:
                        sleep_sec = 15 * (2 ** attempt)  # 15, 30, 60 сек
                        time.sleep(sleep_sec)
                    continue
                # Остальные 4xx — бизнес-ошибки, не ретраим
                if 400 <= e.code < 500:
                    break
                # 5xx — обычный backoff
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except Exception as e:
                # TimeoutError / SocketTimeoutError — ожидаемо в degraded mode → debug
                etype = type(e).__name__
                last_error = f"{etype}: {e}"
                level = logger.debug if "Timeout" in etype else logger.warning
                level("Polza %s attempt %d/%d failed: %s",
                      model, attempt + 1, self.max_retries + 1, last_error)
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        self.failed_calls += 1
        return PolzaResponse(
            success=False,
            error=last_error or "unknown",
            model=model,
            elapsed_sec=time.time() - t0,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Пытается вытащить JSON из markdown-блока или произвольного текста."""
        text = text.strip()
        # Markdown code block
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass
        if "```" in text:
            try:
                start = text.index("```") + 3
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass
        # Brace match
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass
        return None

    def usage_report(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "approx_cost_rub": (self.total_tokens_in + self.total_tokens_out) / 1000 * 0.5,  # очень грубо
        }

    # ─────────── Async версия для параллельного выполнения ───────────

    async def async_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> PolzaResponse:
        """
        Async версия chat() — для параллельного выполнения через asyncio.gather().

        Принципиально идентична sync chat(), но использует aiohttp вместо urllib.
        Никогда не бросает исключений — возвращает PolzaResponse с success=False.
        """
        if not self.is_configured:
            return PolzaResponse(success=False, error="POLZA_API_KEY не задан", model=model)

        if model not in ALLOWED_MODELS:
            logger.warning("Polza: модель %s не в ALLOWED_MODELS", model)

        try:
            import aiohttp
            import asyncio
        except ImportError:
            return PolzaResponse(success=False, error="aiohttp не установлен", model=model)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Optional[str] = None
        t0 = time.time()

        # SSL контекст с certifi (создаём один раз, переиспользуем во всех попытках)
        ssl_context = None
        try:
            import ssl
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass

        for attempt in range(self.max_retries + 1):
            try:
                # Новый connector на каждой попытке (старый закрывается с сессией)
                connector = aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else None
                timeout = aiohttp.ClientTimeout(total=self.timeout, connect=10, sock_read=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status >= 400:
                            text = await resp.text()
                            last_error = f"HTTP {resp.status}: {text[:200]}"
                            level = logger.debug if resp.status in (429, 408, 503, 504) else logger.warning
                            level("Polza async %s attempt %d/%d failed: %s",
                                  model, attempt + 1, self.max_retries + 1, last_error)
                            # 429 (rate limit) РЕТРАИМ с длинным backoff
                            if resp.status == 429:
                                if attempt < self.max_retries:
                                    # 429 → ждём 5, 10, 20 сек
                                    await asyncio.sleep(5 * (2 ** attempt))
                                continue
                            # Остальные 4xx (400, 401, 404, ...) — не ретраим, бизнес-ошибки
                            if 400 <= resp.status < 500:
                                self.failed_calls += 1
                                return PolzaResponse(success=False, error=last_error, model=model)
                            if attempt < self.max_retries:
                                await asyncio.sleep(2 ** attempt)
                            continue

                        data = await resp.json()

                content = ""
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    last_error = "malformed response"
                    continue

                usage = data.get("usage", {})
                tokens_in = int(usage.get("prompt_tokens", 0))
                tokens_out = int(usage.get("completion_tokens", 0))

                self.total_tokens_in += tokens_in
                self.total_tokens_out += tokens_out
                self.total_calls += 1

                parsed = None
                if json_mode:
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        parsed = self._extract_json(content)

                return PolzaResponse(
                    success=True,
                    content=content,
                    parsed_json=parsed,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                    elapsed_sec=time.time() - t0,
                )

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning("Polza async %s attempt %d/%d exception: %s",
                               model, attempt + 1, self.max_retries + 1, last_error)
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        self.failed_calls += 1
        return PolzaResponse(success=False, error=last_error or "unknown", model=model)
