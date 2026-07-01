# MOEX AI Bot

Автономный торговый контур для ArenaGo/MOEX: контейнер запускается, синхронизирует портфель, принимает торговый план, проверяет риск, исполняет market-заявки, хранит `/data/state.json` и пишет `/data/audit.log`.

Multi-agent LLM decision-maker из командной архитектуры подключается через `src/decision/interface.py`. До подключения настоящего агента `src/decision/stub.py` возвращает пустой план, чтобы контейнер можно было безопасно проверить.

Для быстрой навигации по файлам см. `PROJECT_MAP.md`.

## Что уже заложено

- ArenaGo API client: `submit_order`, `positions`, `trades`, `bots`.
- Executor для market-ордеров и подтверждения через ArenaGo.
- `/data/state.json` с atomic checkpoint и reconcile по фактическим позициям.
- Эмуляция стопов и тейков через `StopWatcher`.
- `/healthz` endpoint для проверки состояния контейнера, `/data` и ArenaGo.
- Structured JSON logs и append-only `/data/audit.log`.
- Поставка в одном контейнере через `Dockerfile` в корне проекта.
- Dry-run режим по умолчанию: реальные заявки отправляются только при `TRADING_ENABLED=true`.

## Структура

```text
.
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── src
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── arenago_client.py
│   ├── executor.py
│   ├── state.py
│   ├── logger.py
│   ├── health.py
│   ├── risk.py
│   ├── stop_watcher.py
│   ├── decision
│   │   ├── interface.py
│   │   └── stub.py
│   └── data
│       ├── interface.py
│       └── moex_iss.py
```

## Переменные окружения

- `SANDBOX_API_KEY` — токен ArenaGo. На этапах хакатона подменяется организаторами автоматически.
- `BOT_NAME` — имя портфеля/бота в ArenaGo, по умолчанию `MyBot`.
- `DATA_DIR` — директория постоянного состояния, по умолчанию `/data`.
- `ARENAGO_BASE_URL` — по умолчанию `https://arenago.ru`.
- `HEALTH_HOST` / `HEALTH_PORT` — endpoint health-check, по умолчанию `0.0.0.0:8080`.
- `DECISION_INTERVAL_SECONDS` — период decision cycle, по умолчанию 900 секунд.
- `STOP_WATCH_INTERVAL_SECONDS` — базовый интервал StopWatcher, сейчас используется как настройка для следующего шага с отдельным watcher-loop.
- `TRADING_ENABLED` — если `true`, executor и StopWatcher отправляют реальные заявки в ArenaGo. По умолчанию `false`, события пишутся как dry-run.

## Запуск

```bash
export SANDBOX_API_KEY=...
export BOT_NAME=...
python -m src.main
```

Docker:

```bash
docker build -t moex-ai-bot .
docker run --rm -e SANDBOX_API_KEY=... -e BOT_NAME=... -v /tmp/moex-data:/data moex-ai-bot
```

Для реальной торговли:

```bash
export TRADING_ENABLED=true
python -m src.main
```

## Важные ограничения

- Торговый контур умеет исполнять заявки, но текущий default decision-provider безопасно возвращает пустой план. Боевой LLM/LangGraph provider должен реализовать `DecisionProvider`.
- По требованиям организаторов поставка предполагается в одном контейнере: используем только `Dockerfile` в корне проекта, без Docker Compose и связок из нескольких контейнеров.
- Модели в этом инфраструктурном слое не исполняются. Они подключаются отдельным decision/data слоем для обработки информации и формирования входа в executor.
- Кнопки включения/выключения бота через `ENABLE_OPS_BUTTONS=true` доступны только на первом этапе и не заменяют recovery/checkpoint в `/data`.
- StopWatcher закрывает позиции market-ордером по последней цене MOEX ISS. ArenaGo остается источником истины по фактическому исполнению.
- Количество в `size_lots` передается в ArenaGo как `quantity`, согласно API чемпионата.

## Куда подключать командные части

| Часть | Куда вставлять | Что должна вернуть |
|---|---|---|
| LLM/LangGraph decision-maker | `src/decision/interface.py`, замена `StubDecisionProvider` в `src/main.py` | `DecisionPlan` с ордерами |
| Рыночные данные/AlgoPack/новости | `src/data/` | Данные для decision-provider и цены для executor/stops |
| Risk-лимиты | `src/risk.py` | `ValidationResult` перед отправкой заявки |
| Stop/take logic | `src/stop_watcher.py` и `StopSpec` в `src/schemas.py` | Закрывающие market-заявки |
| ArenaGo execution | `src/arenago_client.py`, `src/executor.py` | Реальные `submit_order` при `TRADING_ENABLED=true` |

## Что уже закрыто по заданию

| Требование | Где реализовано |
|---|---|
| Dockerfile в корне, один контейнер | `Dockerfile`, `.dockerignore` |
| ArenaGo API client | `src/arenago_client.py` |
| Исполнение market-ордеров | `src/executor.py` |
| `/data` checkpoint/recovery | `src/state.py` |
| Reconcile с ArenaGo | `src/state.py`, `src/main.py`, `src/executor.py` |
| Эмуляция стопов/тейков | `src/stop_watcher.py` |
| Health-check | `src/health.py` |
| Structured logs + audit | `src/logger.py` |
| Deterministic risk gate | `src/risk.py` |
| Точка подключения LLM decision-maker | `src/decision/interface.py` |
| Точка подключения data pipeline | `src/data/interface.py` |

## Инструкция запуска

### 1. Локальный dry-run

Dry-run поднимает бота, проверяет ArenaGo, пишет state/audit, но не отправляет заявки.

```bash
export SANDBOX_API_KEY="<arena_go_token>"
export BOT_NAME="<arena_go_bot_name>"
export DATA_DIR="/tmp/moex-ai-data"
export TRADING_ENABLED=false

python -m src.main
```

Во втором терминале:

```bash
curl -i http://127.0.0.1:8080/healthz
ls -la /tmp/moex-ai-data
tail -n 20 /tmp/moex-ai-data/audit.log
```

### 2. Локальный запуск с реальным исполнением

Включать только когда decision-provider уже возвращает нужный `DecisionPlan`.

```bash
export SANDBOX_API_KEY="<arena_go_token>"
export BOT_NAME="<arena_go_bot_name>"
export DATA_DIR="/tmp/moex-ai-data"
export TRADING_ENABLED=true

python -m src.main
```

### 3. Docker dry-run

```bash
docker build -t moex-ai-bot .

docker run --rm \
  -e SANDBOX_API_KEY="<arena_go_token>" \
  -e BOT_NAME="<arena_go_bot_name>" \
  -e TRADING_ENABLED=false \
  -p 8080:8080 \
  -v /tmp/moex-ai-data:/data \
  moex-ai-bot
```

### 4. Docker real trading

```bash
docker run --rm \
  -e SANDBOX_API_KEY="<arena_go_token>" \
  -e BOT_NAME="<arena_go_bot_name>" \
  -e TRADING_ENABLED=true \
  -p 8080:8080 \
  -v /tmp/moex-ai-data:/data \
  moex-ai-bot
```
