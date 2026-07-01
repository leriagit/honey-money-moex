# Project Map

Это карта рабочего торгового контура. Проект запускает контейнер, держит состояние в `/data`, получает состояние портфеля из ArenaGo, принимает торговый план, проверяет риск, исполняет market-заявки и пишет audit.

Multi-agent LLM-логика из командных документов подключается через `src/decision`. Расширенные данные, новости, ML-сигналы и RAG подключаются через `src/data`.

## С чего читать

1. `src/main.py` — старт контейнера и общий runtime-flow.
2. `src/schemas.py` — общие объекты: решение, заявка, позиция, состояние.
3. `src/decision/interface.py` — место подключения будущего LLM/LangGraph decision-maker.
4. `src/executor.py` — превращает решение в реальные заявки ArenaGo.
5. `src/state.py` и `src/stop_watcher.py` — восстановление после рестарта и виртуальные стопы.

## Поток данных

```text
ArenaGo positions/trades/cash
  -> StateStore.reconcile_from_snapshots()
  -> DecisionProvider.decide(context)
  -> DecisionPlan
  -> Executor.execute_plan()
  -> Risk.validate()
  -> ArenaGoClient.submit_order() if TRADING_ENABLED=true
  -> /data/state.json + /data/audit.log
```

Параллельно:

```text
/data/state.json active_stops
  -> StopWatcher
  -> MarketDataProvider.get_last_price()
  -> ArenaGoClient.submit_order(close)
  -> /data/audit.log
```

## Файлы

| Файл | Зачем нужен | Что получает | Что отдает дальше |
|---|---|---|---|
| `src/main.py` | Запускает бота и связывает модули | Env-настройки, ArenaGo snapshots | Decision cycle, stop watcher, health server |
| `src/config.py` | Читает настройки контейнера | `SANDBOX_API_KEY`, `BOT_NAME`, `DATA_DIR`, лимиты | `Settings` |
| `src/schemas.py` | Единый язык объектов проекта | Поля API и внутренние поля | Pydantic-модели для всех модулей |
| `src/arenago_client.py` | Общается с ArenaGo API | HTTP endpoints ArenaGo | `BotInfo`, `Position`, `Trade`, `OrderResponse` |
| `src/decision/interface.py` | Контракт для будущего decision-maker | `DecisionContext` | `DecisionPlan` |
| `src/decision/stub.py` | Безопасный provider до подключения LLM | `DecisionContext` | Пустой `DecisionPlan`, без сделок |
| `src/executor.py` | Исполняет торговый план | `DecisionPlan`, позиции, цена, risk gate | Market-заявки в ArenaGo, `ExecutionResult`, audit |
| `src/risk.py` | Детерминированно проверяет заявку | Заявка, cash, state, позиции | `allowed/reason` |
| `src/state.py` | Хранит checkpoint в `/data` | ArenaGo snapshots, стопы | `/data/state.json` |
| `src/stop_watcher.py` | Эмулирует stop-loss/take-profit | `active_stops`, последняя цена | Закрывающая market-заявка |
| `src/data/interface.py` | Контракт источника цены | Тикер | Последняя цена или `None` |
| `src/data/moex_iss.py` | Базовая цена из MOEX ISS | Тикер | Последняя доступная цена |
| `src/logger.py` | Пишет operational log и audit | Runtime-события | stdout, `/data/logs/app.jsonl`, `/data/audit.log` |
| `src/health.py` | Health-check контейнера | StateStore, ArenaGo client | `/healthz` |

## Соответствие выбранной архитектуре

| Требование из документов | Где отражено сейчас | Статус |
|---|---|---|
| ArenaGo API client | `src/arenago_client.py` | Есть |
| Executor market-ордеров | `src/executor.py` | Есть |
| `/data` checkpoint/recovery | `src/state.py` | База есть, можно расширять |
| StopWatcher | `src/stop_watcher.py` | Есть |
| Structured logs + audit | `src/logger.py` | Есть |
| Health-check | `src/health.py` | Есть |
| Deterministic risk gate | `src/risk.py` | База есть, лимиты можно расширять |
| LangGraph/LLM decision system | `src/decision/interface.py` | Только точка подключения |
| ML/news/RAG data pipeline | `src/data/interface.py` | Только точка подключения |

## Что важно не перепутать

- `src/decision/stub.py` не является стратегией. Он нужен, чтобы контейнер запускался до подключения настоящего агента.
- `src/data/moex_iss.py` не является полноценным data pipeline. Сейчас он нужен executor-у и watcher-у только для reference price.
- ArenaGo считается источником истины по cash, позициям и сделкам.
- Поставка проекта идет одним контейнером через `Dockerfile`; `.yml`, Docker Compose и несколько контейнеров не нужны.
- Реальные заявки отправляются только при `TRADING_ENABLED=true`. Без этого executor пишет dry-run события в audit.

## Точки интеграции для команды

| Кто подключает | Файл | Что сделать |
|---|---|---|
| LLM/decision | `src/decision/interface.py` | Реализовать provider с методом `decide(context) -> DecisionPlan` |
| LLM/decision | `src/main.py` | Заменить `StubDecisionProvider()` на боевой provider |
| Data/AlgoPack/news | `src/data/` | Добавить источники данных и передать их в decision-provider |
| Risk/strategy | `src/risk.py` | Расширить deterministic gates: drawdown, sector cap, cooldown |
| Execution | `src/executor.py` | Не менять контракт: сюда должен приходить готовый `DecisionPlan` |

## Архитектура decision-слоя (после рефакторинга)

Раньше вся логика жила в одном файле `honey_money_provider.py` на 936 строк
(god-object с двумя случайно продублированными методами). Теперь слой разбит
на модули с одной ответственностью; публичный контракт `decide(context) ->
DecisionPlan` не изменился.

| Файл | Ответственность |
|---|---|
| `src/decision/honey_money_provider.py` | Тонкий оркестратор: связывает компоненты, не содержит бизнес-деталей |
| `src/decision/strategy_params.py` | Единый источник всех порогов/лимитов (`StrategyParams`), переопределяется из env |
| `src/decision/market_context.py` | Сеть + кэш: свечи MOEX, RSS-новости, IMOEX, live USD/RUB, Brent, realized_vol (с фолбэком на `macro_priors.yaml`) |
| `src/decision/signals.py` | Технические индикаторы + ML prob_up |
| `src/decision/order_mapper.py` | Торговые guard-ы (anti-wash, min-hold, anti-averaging) и сайзинг |

## Статус модулей `src/risk_extra/`

| Модуль | Статус | Где подключён |
|---|---|---|
| `crisis_gate.py` | ✅ подключён | `HoneyMoneyDecisionProvider._apply_crisis_gate` — при обвале IMOEX отбрасывает открывающие ордера (закрытия пропускает) |
| `position_sizer.py` (kelly) | покрыт тестами, готов к включению в сайзинг | `tests/test_risk_extra.py` |
| `stop_loss.py` (`StopLossManager`) | НЕ включается в live-цикл намеренно | дублировал бы `StopWatcher` → риск двойного закрытия. Используется как backtest-инструмент |

## Тесты

`tests/` (pytest) покрывают order_mapper, risk gate, risk_extra, конфиг и сквозной
`decide()`. Запуск: `python -m pytest tests/`.
