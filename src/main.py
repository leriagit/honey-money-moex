"""Точка входа контейнера."""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from threading import Event

import uvicorn

from src.arenago_client import ArenaGoClient
from src.config import load_settings
from src.data import MoexISSMarketDataProvider
from src.decision import (
    DecisionContext,
    DecisionProvider,
    HoneyMoneyDecisionProvider,
    StubDecisionProvider,
)
from src.decision.strategy_params import StrategyParams
from src.executor import Executor
from src.health import HealthService, create_app
from src.logger import setup_logging, write_audit_event
from src.risk import PreTradeValidator, RiskConfig
from src.state import StateStore
from src.stop_watcher import StopWatcher


def main() -> None:
    """Собирает зависимости и запускает основной процесс бота."""

    settings = load_settings(require_api_key=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(settings.data_dir)
    stop_event = Event()
    _install_signal_handlers(stop_event)

    client = ArenaGoClient(
        api_key=settings.arenago_api_key,
        base_url=settings.arenago_base_url,
        bot_name=settings.bot_name,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.request_max_retries,
    )
    # MOEX ISS бывает медленный — даём 20s connect / 30s read, 3 retry с backoff.
    # При полном отказе ISS get_last_price() отдаёт stale-цену из кеша
    # (TTL=10 мин) — это позволяет stop_watcher продолжать работу.
    market_data = MoexISSMarketDataProvider(
        timeout_seconds=30.0,
        connect_timeout=20.0,
        max_retries=max(3, settings.request_max_retries),
        cache_ttl_seconds=600.0,
    )
    state_store = StateStore(settings.data_dir)
    risk = PreTradeValidator(
        RiskConfig(
            max_daily_trades=settings.max_daily_trades,
            max_order_cash_share=settings.max_order_cash_share,
            min_cash_reserve=settings.min_cash_reserve,
            allow_shorts=settings.allow_shorts,
        )
    )
    executor = Executor(
        client,
        state_store,
        risk,
        market_data,
        settings.data_dir,
        logger,
        trading_enabled=settings.trading_enabled,
    )
    # Боевой decision-provider команды honey_money_bot:
    # LLM-граф (9 агентов через Polza.ai) + LightGBM × MacroContext overlay
    # + VolumeGuard + CrisisGate. Все компоненты имеют graceful fallback.
    # Если нужно откатиться на пустой план — раскомментируй Stub:
    # decision_provider = StubDecisionProvider()
    #
    # Все торговые пороги собраны в StrategyParams (единый источник правды,
    # переопределяется из окружения) — это устраняет прежний рассинхрон, когда
    # stop-loss задавался одновременно как 0.03 и 0.024 в разных файлах.
    strategy_params = StrategyParams.from_env()
    decision_provider = HoneyMoneyDecisionProvider(
        data_dir=settings.data_dir,
        model_path="models/lightgbm_v1.txt",
        macro_priors_path="config/macro_priors.yaml",
        params=strategy_params,
        enable_llm=True,
        enable_news=True,  # RSS keyless источники (Reuters, NYT, WSJ, FT, MOEX news, РБК, ...)
        enable_crisis_gate=True,
        market_data=market_data,  # шарим кеш цен → stop_watcher переживает падение ISS
    )
    stop_watcher = StopWatcher(
        client,
        state_store,
        market_data,
        settings.data_dir,
        logger,
        trading_enabled=settings.trading_enabled,
    )
    _start_stop_watcher_loop(
        stop_watcher=stop_watcher,
        interval_seconds=settings.stop_watch_interval_seconds,
        stop_event=stop_event,
        logger=logger,
    )

    health_service = HealthService(
        client,
        state_store,
        settings.data_dir,
        trading_enabled=settings.trading_enabled,
        decision_provider_name=decision_provider.__class__.__name__,
    )
    health_app = create_app(health_service)
    _start_health_server(health_app, settings.health_host, settings.health_port)

    startup_event = {
        "bot": settings.bot_name,
        "trading_enabled": settings.trading_enabled,
        "decision_provider": decision_provider.__class__.__name__,
    }
    logger.info("bot_started", extra={"event": startup_event})
    write_audit_event(settings.data_dir, "bot_started", startup_event)

    try:
        while not stop_event.is_set():
            try:
                _run_decision_cycle(client, state_store, decision_provider, executor, settings.data_dir)
            except Exception as error:
                state_store.record_error(str(error))
                event = {"error": str(error)}
                write_audit_event(settings.data_dir, "decision_cycle_failed", event)
                logger.exception("decision_cycle_failed", extra={"event": event})
            stop_event.wait(settings.decision_interval_seconds)
    finally:
        write_audit_event(settings.data_dir, "bot_stopped", {"bot": settings.bot_name})
        logger.info("bot_stopped", extra={"event": {"bot": settings.bot_name}})
        client.close()
        market_data.close()


def _run_decision_cycle(
    client: ArenaGoClient,
    state_store: StateStore,
    decision_provider: DecisionProvider,
    executor: Executor,
    data_dir: Path,
) -> None:
    """Выполняет один decision-cycle: reconcile, decide, audit, execute."""

    cash = client.get_cash_balance()
    positions = client.get_positions()
    trades_today = client.get_trades()
    state = state_store.reconcile_from_snapshots(cash, positions, trades_today)
    context = DecisionContext(state=state, positions=positions, trades_today=trades_today)
    plan = decision_provider.decide(context)
    write_audit_event(data_dir, "decision_plan", plan.model_dump(mode="json"))
    executor.execute_plan(plan)


def _install_signal_handlers(stop_event: Event) -> None:
    """Устанавливает обработчики SIGTERM/SIGINT для graceful shutdown."""

    def handle_signal(signum: int, _frame: object) -> None:
        """Переводит процесс в режим штатной остановки."""

        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def _start_health_server(app, host: str, port: int) -> None:
    """Запускает health endpoint в отдельном daemon-thread."""

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()


def _start_stop_watcher_loop(
    stop_watcher: StopWatcher,
    interval_seconds: float,
    stop_event: Event,
    logger,
) -> None:
    """Запускает отдельный цикл проверки виртуальных стопов."""

    def loop() -> None:
        """Периодически запускает StopWatcher до остановки процесса."""

        while not stop_event.is_set():
            try:
                stop_watcher.run_once()
            except Exception as error:
                logger.exception("stop_watcher_failed", extra={"event": {"error": str(error)}})
            stop_event.wait(interval_seconds)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


if __name__ == "__main__":
    main()
