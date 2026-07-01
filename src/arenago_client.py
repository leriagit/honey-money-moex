"""Клиент ArenaGo API."""

# Закрытая часть торгового контура: портфель, позиции, сделки и отправка
# market-заявок в ArenaGo проходят через этот модуль.

from __future__ import annotations

from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.schemas import BotInfo, OrderRequest, OrderResponse, OrderSide, Position, Trade


class ArenaGoError(RuntimeError):
    """Базовое исключение ArenaGo-клиента."""

    pass


class ArenaGoTransientError(ArenaGoError):
    """Временная ошибка, для которой допустим retry."""

    pass


class ArenaGoRejectedError(ArenaGoError):
    """Ошибка отказа API или невалидного ответа ArenaGo."""

    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        """Создает исключение с человекочитаемой причиной и исходным payload."""

        super().__init__(message)
        self.payload = payload or {}


class ArenaGoClient:
    """Синхронный клиент торгового API ArenaGo."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        bot_name: str,
        timeout_seconds: float = 10,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        """Инициализирует HTTP-клиент с авторизацией и настройками retry."""

        if not api_key:
            raise ValueError("SANDBOX_API_KEY is not set")

        self.bot_name = bot_name
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    def close(self) -> None:
        """Закрывает HTTP-соединения клиента."""

        self._client.close()

    def _request(self, method: str, path: str, json_payload: dict[str, Any] | None = None) -> Any:
        """Выполняет HTTP-запрос к ArenaGo и нормализует ошибки API."""

        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, max=5),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, ArenaGoTransientError)),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                response = self._client.request(method, path, json=json_payload)
                if response.status_code >= 500:
                    raise ArenaGoTransientError(f"ArenaGo HTTP {response.status_code}: {response.text}")
                if response.status_code >= 400:
                    raise ArenaGoRejectedError(f"ArenaGo HTTP {response.status_code}: {response.text}")

                try:
                    data = response.json()
                except ValueError as error:
                    raise ArenaGoRejectedError(
                        "ArenaGo returned non-JSON response",
                        payload={"text": response.text},
                    ) from error
                if isinstance(data, dict) and data.get("error"):
                    raise ArenaGoRejectedError(str(data["error"]), payload=data)
                return data

        raise ArenaGoTransientError("ArenaGo request retry loop exited unexpectedly")

    def get_bots(self) -> list[BotInfo]:
        """Получает список портфелей, доступных текущему API-ключу."""

        data = self._request("GET", "/api/bots")
        if not isinstance(data, list):
            raise ArenaGoRejectedError("ArenaGo /api/bots returned non-list payload", payload={"payload": data})
        return [BotInfo.model_validate(item) for item in data]

    def get_bot(self) -> BotInfo:
        """Находит текущий портфель по имени `bot_name`."""

        for bot in self.get_bots():
            if bot.name == self.bot_name:
                return bot
        raise ArenaGoRejectedError(f"Bot '{self.bot_name}' not found")

    def get_cash_balance(self) -> float:
        """Возвращает cash balance текущего портфеля."""

        return self.get_bot().cash_balance

    def get_positions(self) -> list[Position]:
        """Получает фактические открытые позиции текущего портфеля."""

        data = self._request("GET", f"/api/positions/{self.bot_name}")
        if not isinstance(data, list):
            raise ArenaGoRejectedError("ArenaGo positions returned non-list payload", payload={"payload": data})
        return [Position.model_validate(item) for item in data]

    def get_trades(self) -> list[Trade]:
        """Получает сегодняшние сделки текущего портфеля."""

        data = self._request("GET", f"/api/trades/{self.bot_name}")
        if not isinstance(data, list):
            raise ArenaGoRejectedError("ArenaGo trades returned non-list payload", payload={"payload": data})
        return [Trade.model_validate(item) for item in data]

    def submit_order(self, direction: OrderSide, secid: str, quantity: int) -> OrderResponse:
        """Отправляет рыночную заявку в ArenaGo."""

        request = OrderRequest(
            direction=direction,
            secid=secid,
            quantity=quantity,
            bot=self.bot_name,
        )
        data = self._request("POST", "/api/submit_order", json_payload=request.model_dump(mode="json"))
        if not isinstance(data, dict):
            raise ArenaGoRejectedError("ArenaGo submit_order returned non-object payload", payload={"payload": data})
        response = OrderResponse.model_validate(data)
        if not response.success:
            raise ArenaGoRejectedError(response.message or "ArenaGo order rejected", payload=data)
        return response
