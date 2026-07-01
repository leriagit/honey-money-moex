# honey money — ИИ-агент для торгов на MOEX

> Автономный мульти-агентный торговый бот для тестового контура **ArenaGo**.
> 9 LLM-ролей через open-source модели + LightGBM ML + макро-overlay аналитика.
> Команда **team-05** ("honey money"), MOEX AI Hackathon 2026.

---

## TL;DR

**В одном абзаце:** honey money — это автономный торговый агент, который принимает решения раз в 10 минут на основе ансамбля из трёх независимых источников сигналов: **(1)** мульти-агентный LLM-граф из 9 ролей через polza.ai (NewsAnalyst, Bull/Bear Researchers, Risk Officer, Portfolio Manager), **(2)** обученная LightGBM-модель на 5 годах истории MOEX, **(3)** макро-overlay с приоритетами от человека-аналитика. Стратегия — **mean-reversion + news-driven shorts** в bear-режиме рынка. Главная инженерная фишка: **5-уровневый graceful fallback**, благодаря которому отказ любого компонента (LLM rate-limit, MOEX ISS timeout, отсутствие новостей) не останавливает торговлю.

**В чём наше преимущество:**

| Что | Как мы это решаем |
|---|---|
| LLM делают overconfident-ставки | Bull/Bear дебат + жёсткое правило `news > ML` |
| Один плохой ордер может выжечь портфель | Hard cap 100К ₽ (20% от стартового капитала) на сделку |
| LLM/API могут упасть в любой момент | Rule-based fallback в каждом из 9 агентов |
| MOEX ISS бывает медленный/недоступен | Кеш цен с TTL=10 мин + stale-fallback |
| Whipsaw: stop-loss → откупаемся → опять стоп | Cooldown 5 мин после стопа + anti-averaging-down/up |

---

## Quick Start

### Локальный запуск

```bash
git clone https://hackaton.gitlab.yandexcloud.net/hackathon/team-05.git
cd team-05
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # заполнить SANDBOX_API_KEY и POLZA_API_KEY
PYTHONPATH=. python -m src.main
```

После запуска бот:
- через 1-2 минуты сделает первый decision-cycle
- логирует JSON-события в stdout
- открывает read-only health-endpoint на `http://localhost:8080/healthz`
- пишет state в `./data/` (или в `/data/` в Docker)

### Через Docker

```bash
docker build -t honey_money .
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/data \
  -p 8080:8080 \
  honey_money
```

### На сервере organizers (этап 2)

Bezусловный авто-деплой через GitLab CI. От нас нужен только зелёный pipeline на `main` ветке.
Шаги для ручного запуска (этап 1):
1. **Builds → Pipelines → New pipeline** на ветке `main`, переменная `ENABLE_OPS_BUTTONS=yes`
2. Стадия `push` → `push_registry` (собирает Docker image и пушит в их registry)
3. Стадия `deploy` → `deploy_helm` (разворачивает на их Kubernetes)
4. Стадия `ops` → `storage_enable` → `bot_start`
5. Логи: **Monitor → Dashboards**

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│  СЛОЙ 1. Источники данных (раз в 10 мин)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ MOEX ISS    │  │ ArenaGo REST │  │ 9 RSS    │  │ macro_priors │  │
│  │ aiomoex     │  │ позиции/cash │  │ новости  │  │ .yaml        │  │
│  └──────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘  │
└─────────┼────────────────┼────────────────┼────────────────┼─────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  СЛОЙ 2. Преобразование сигналов                                    │
│  • features.py — 50 фичей (RSI/MACD/ATR/momentum 1h/4h/24h)         │
│  • ml_signal.py — LightGBM prob_up по тикеру (AUC test 0.560)       │
│  • indicators.py — технический bias (long/short/neutral)            │
│  • news_aggregator.py — дедупликация заголовков по hash             │
│  • macro_context.py — logit-сдвиги по 24 макро-флагам               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  СЛОЙ 3. Multi-agent LLM граф (9 ролей)                             │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐ ┌────────┐  │
│  │ Regime   │ │ Technical │ │ News     │ │ Fundamental│ │ Pair   │  │
│  │ Analyst  │ │ Analyst   │ │ Analyst  │ │ Analyst    │ │ Analyst│  │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └─────┬──────┘ └───┬────┘  │
│       └──────────────┴────┬───────┴─────────────┴───────────┘       │
│                           ▼                                         │
│            ┌──────────────────────────────────┐                     │
│            │  Bull Researcher ⟷ Bear Researcher  (debate)           │
│            └──────────────┬───────────────────┘                     │
│                           ▼                                         │
│                  ┌─────────────────┐                                │
│                  │  Risk Officer   │  (детерминированные gates)     │
│                  └────────┬────────┘                                │
│                           ▼                                         │
│                ┌────────────────────┐                               │
│                │  Portfolio Manager │  (финальный JSON ордеров)     │
│                └─────────┬──────────┘                               │
└──────────────────────────┼──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  СЛОЙ 4. Risk overlay (force-exit, имеет veto над LLM)              │
│  • stop-loss -3% от avg_price → SELL_ALL                            │
│  • take-profit +2% → SELL_HALF                                      │
│  • trailing-stop 1% после +1.5%                                     │
│  • hard drawdown -5% от peak → SELL_ALL                             │
│  • cooldown 5 мин после стопа (anti-whipsaw)                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  СЛОЙ 5. Executor → ArenaGo                                         │
│  • Hard cap 100К ₽ на ордер (20% стартового капитала)               │
│  • Anti-averaging-down/up, MIN_ORDER 5К ₽                           │
│  • VolumeGuard следит за оборотом ≥10М ₽ (требование ТЗ)            │
│  • Audit-лог JSON в /data/audit/                                    │
└─────────────────────────────────────────────────────────────────────┘
```

Каждый слой имеет **graceful fallback**:
- ML недоступна → BaselineMLProvider возвращает prob_up=0.5
- LLM rate-limit → каждый из 9 агентов имеет rule-based fallback
- Новости не парсятся → агенты работают только на технике
- MOEX ISS timeout → отдаём цену из кеша

**Итог: бот не останавливается никогда.**

---

## Multi-agent LLM граф (наша главная фишка)

### Зачем 9 ролей, а не один монолитный LLM-call?

Известная проблема LLM в трейдинге — **overconfidence**: модель уверенно даёт BUY на новость, которую неправильно интерпретировала. Классическое решение в исследованиях ([TradingAgents, arXiv 2402.18485]) — разделить роли:

1. **Аналитики** только описывают что видят (без рекомендаций)
2. **Researchers** аргументируют (Bull vs Bear) и спорят
3. **Risk Officer** срезает overconfident-идеи через жёсткие gates
4. **Portfolio Manager** синтезирует финальное решение

Этот pipeline даёт **разделение перспектив** — Bull-агент специально ищет позитивные сигналы, Bear — негативные, и Portfolio Manager делает осознанный выбор между ними.

### Все 9 ролей в нашей реализации

| № | Агент | Модель / Источник | Что возвращает | Fallback |
|---|---|---|---|---|
| 1 | Regime Analyst | rule-based | `regime`, `volatility`, `bias` | always |
| 2 | Technical Analyst | rule-based + 50 фичей | per-ticker `bias` + `strength` | always |
| 3 | **News Analyst** | mistral-large | per-ticker `sentiment ±1`, `narrative` | neutral baseline |
| 4 | Fundamental Analyst | macro_priors.yaml | per-ticker `bias` по 24 флагам | always |
| 5 | Pair Analyst | rule-based корреляция | cointegration spreads | always |
| 6 | **Bull Researcher** | mistral-large, T=0.4 | ТОП-3 long-идеи + conviction | tech-derived longs |
| 7 | **Bear Researcher** | mistral-large, T=0.4 | ТОП-3 short-идеи + macro_risks | tech-derived shorts |
| 8 | Risk Officer | rule-based gates | approved_longs, approved_shorts, vetoed | always (детерминирован) |
| 9 | **Portfolio Manager** | mistral-large, T=0.1 | финальный JSON ордеров | ML-based fallback |

**Активные LLM-агенты в проде:** 4 (выделены жирным). Остальные 5 на rule-based — это сознательное решение для скорости и устойчивости (мы тестировали все 9 на LLM — на 4 LLM получается ~50-60 сек на цикл, что вписывается в 10-минутный интервал с большим запасом).

### Ключевое правило: news_sentiment > ML prob_up

Это **жёстко прописано** в system-prompt Portfolio Manager:

```
ПРАВИЛО ПРИОРИТЕТА: NEWS sentiment >= ML prob_up.
• news_sentiment < -0.3 → НЕ открывай long даже при высоком prob_up
• news_sentiment > +0.3 → НЕ открывай short даже при низком prob_up
```

**Реальный пример из работы бота** (26 мая, 09:48):
- LightGBM по GMKN: `prob_up = 0.60` → ML говорит "BUY"
- NewsAnalyst: `sentiment = -0.90` (новость: ГМК отказался от дивидендов)
- Portfolio Manager: **открыл SHORT** на GMKN (49К ₽), приоритет новостей перебил ML
- За 12 часов: GMKN +1.20% против нас, но не превысил stop (+2.32%), позиция жива и работает

---

## Стратегия торгов

### Базовая идея

**Mean-reversion + news-driven shorts** в bearish-режиме. Конкретно:

1. **Не пытаемся ловить тренд** — слишком много шума на 10-минутном таймфрейме
2. **Открываем short на негативную новость + bearish-сектор** (например, отмена дивидендов металлургов)
3. **Открываем long только при сильном bullish-новостном фоне + ML-подтверждении**
4. **Быстро фиксируем прибыль** (+2% take-profit) — на bear-рынке не жадничаем
5. **Жёстко режем убытки** (-3% stop-loss) — никаких "усреднений на падении"

### Параметры стратегии

| Параметр | Значение | Обоснование |
|---|---:|---|
| `DECISION_INTERVAL` | 600 сек (10 мин) | Баланс свежести / стоимости LLM-вызовов |
| `MIN_HOLD_SECONDS` | 300 сек (5 мин) | Anti-whipsaw, защита от дёрганий на каждый тик |
| `MIN_ORDER_RUB` | 5 000 ₽ | Отсечь mosquito-ордера на 1 лот |
| `MAX_ORDER_RUB` (hard cap) | **100 000 ₽** | 20% от 500К стартового капитала на сделку |
| `vol_multiplier` | 0.5 … 1.5 | Equal-risk weighting между позициями |
| `default_take_pct` | +2% | Быстрая фиксация на bear-рынке |
| `default_stop_pct` | −3% | Стандарт по брифу |
| `trailing_start_pct` | +1.5% | Перевод в безубыток после хорошего движения |
| `cooldown_seconds` | 300 (5 мин) | Блок повторного BUY после стопа |
| `MAX_POSITION_PCT` | 20% | Risk Officer veto'ит идеи > 20% от портфеля |

### Risk Management — что нас отличает

```python
# src/decision/honey_money_provider.py — фрагменты ключевой логики

# 1. Hard cap: даже если cash раздут от шорт-proceeds — макс 100К ₽
STARTING_CAPITAL_RUB = 500_000.0
HARD_MAX_ORDER_RUB = STARTING_CAPITAL_RUB * self.max_position_pct  # = 100К ₽

if size_rub > HARD_MAX_ORDER_RUB:
    size_rub = HARD_MAX_ORDER_RUB

# 2. Anti-averaging-down (для long) и anti-averaging-up (для short)
if current_qty > 0 and ref_price <= avg_price * (1.0 + ANTI_AVG_THRESHOLD):
    continue  # не доливаемся в позицию, которая ушла в минус

# 3. Volatility-targeted position sizing
vol_multiplier = target_vol / ticker_vol
vol_multiplier = max(0.5, min(1.5, vol_multiplier))  # cap 1.5×
```

### Что бот сознательно НЕ делает

| Что | Почему |
|---|---|
| ❌ Wash trading (встречные сделки для оборота) | ТЗ явно запрещает → дисквалификация |
| ❌ HFT, scalping | Не наша игра, ставим на качество решений |
| ❌ Closed-source LLM (GPT-4, Claude, Gemini) | ТЗ требует open-source лицензии |
| ❌ Усреднение убыточной позиции | Anti-averaging guard в `_decide_size()` |
| ❌ Открытие позиции без явного позитивного news/ML сигнала | Strict mode по умолчанию |
| ❌ Управление портфелем извне на этапе 2 | `src/health.py` имеет только GET-эндпоинты |

---

## Технологический стек и обоснование выбора

| Категория | Выбор | Почему именно это |
|---|---|---|
| **Язык** | Python 3.12 | Самый быстрый цикл разработки, богатая экосистема ML/finance |
| **ML-модель** | LightGBM | Inference 1 мс, RAM <200 MB, лучше XGBoost на табличных данных |
| **LLM провайдер** | polza.ai | Единый API ко всем open-source моделям, free-tier 6000 ₽ |
| **LLM модель (active)** | Mistral Large | Apache 2.0, нет rate-limit от DeepInfra, хорошо знает русский |
| **LLM модель (reserve)** | DeepSeek-V3 | MIT-лицензия, сильное reasoning, fallback при недоступности Mistral |
| **MOEX данные** | aiomoex | Async-batch, нативная поддержка ISS, без ключа |
| **HTTP-клиент** | httpx + tenacity | Async/sync, встроенный retry, лучше requests |
| **Schema validation** | Pydantic v2 | Type-safe DTO между компонентами, быстрый serialize/deserialize |
| **Health endpoint** | FastAPI + uvicorn | Минимальный overhead, async-нативный |
| **News parsing** | feedparser + lxml | RSS из 9 источников (Reuters, NYT, WSJ, FT, FoxNews, IEA, WaPo, Investing, РБК) |
| **Persistence** | JSON files в /data | Простота, никаких внешних БД — снижение surface для отказов |
| **Контейнеризация** | Docker (python:3.12-slim) | Стандарт хакатона, минимальный образ ~400 MB |
| **CI/CD** | GitLab CI на их инфра | По требованию организаторов |

### Почему именно polza.ai, а не прямой API?

1. **Один ключ ко всем моделям** — переключение Mistral ↔ DeepSeek через смену строки, не переписывая клиент
2. **Биллинг в рублях** через корпоративный аккаунт хакатона (6000 ₽ free-tier)
3. **OpenAI-compatible API** — наш `polza_client.py` работает как с любым OpenAI-совместимым endpoint

### Почему LightGBM, а не глубокая нейросеть?

1. **Tabular data wins** — на бирже фичи структурированы, gradient boosting обычно лучше DNN
2. **Inference 1 мс** — критично для 10-минутного цикла на 20 тикерах
3. **Маленький артефакт** — `models/lightgbm_v1.txt` всего 200 KB, легко версионируется в git
4. **Интерпретируемость** — feature importance даёт понимание что работает

---

## Соответствие требованиям ТЗ

### Правила соревнования (дисквалификация при нарушении)

| Требование | Статус | Где доказательство |
|---|:---:|---|
| Только open-source модели со свободной коммерческой лицензией | ✅ | `src/agents/polza_client.py:ALLOWED_MODELS = {"deepseek-chat", "mistral-large"}` |
| Никакого удалённого управления портфелем на этапе 2 | ✅ | `src/health.py` — только `GET /healthz` (read-only) |
| Все используемые библиотеки — free commercial use | ✅ | См. таблицу лицензий ниже |

### Правила торгов (штраф 70 баллов при нарушении)

| Требование | Статус | Где доказательство |
|---|:---:|---|
| Турновер ≥ 10 000 000 ₽ на этапе 2 | ✅ | `src/data/volume_guard.py` отслеживает + Portfolio Manager увеличивает активность при отставании |
| Никакого wash trading (видно экспертам) | ✅ | MIN_HOLD_SECONDS=5 мин, anti-averaging, нет встречных одновременных ордеров |
| Воспроизводимость модели (веса/код) | ✅ | `models/lightgbm_v1.txt` в репо, обучение в `scripts/train_lightgbm.py` |

### Технические требования

| Требование | Статус | Где |
|---|:---:|---|
| Dockerfile в корне | ✅ | `./Dockerfile` |
| Чтение `SANDBOX_API_KEY` из env | ✅ | `src/config.py:_str_env("SANDBOX_API_KEY")` |
| Persistence в `/data` | ✅ | `Dockerfile: mkdir -p /data && chmod 777 /data` + `ENV DATA_DIR=/data` |
| Все 20 тикеров из ТЗ | ✅ | `src/decision/honey_money_provider.py:TICKERS_FROM_TZ` (LKOH, SBER, ROSN, GAZP, VTBR, YDEX, PLZL, T, NVTK, X5, GMKN, MGNT, ALRS, AFLT, CHMF, NLMK, MOEX, SNGSP, MTSS, PIKK) |
| Поддержка `ENABLE_OPS_BUTTONS=yes` | ✅ | `.gitlab-ci.yml:rules: $ENABLE_OPS_BUTTONS == "yes"` |

### Лицензии используемых моделей и библиотек

| Компонент | Лицензия | Коммерческое использование |
|---|---|:---:|
| Mistral Large (через Polza) | Apache 2.0 | ✅ |
| DeepSeek-V3 (резерв) | MIT | ✅ |
| LightGBM | MIT | ✅ |
| Pandas | BSD-3-Clause | ✅ |
| NumPy | BSD-3-Clause | ✅ |
| Pydantic | MIT | ✅ |
| httpx | BSD-3-Clause | ✅ |
| aiomoex | MIT | ✅ |
| FastAPI | MIT | ✅ |
| feedparser | BSD-2-Clause | ✅ |

---

## Структура репозитория

```
moex-main/
├── src/
│   ├── main.py                   # entry-point + loop decision-циклов
│   ├── config.py                 # env-переменные (SANDBOX_API_KEY и др.)
│   ├── arenago_client.py         # ArenaGo REST: submit_order, positions, trades
│   ├── executor.py               # отправка ордеров + pre-trade validation
│   ├── stop_watcher.py           # параллельный loop проверки stop/take/trailing
│   ├── health.py                 # FastAPI /healthz (read-only)
│   ├── state.py                  # StateStore: позиции, средние, audit
│   ├── risk.py                   # PreTradeValidator: max_order_cash_share, и т.д.
│   ├── schemas.py                # Pydantic-модели (OrderRequest, Position, ...)
│   │
│   ├── agents/                   # ─── Multi-agent LLM граф ───
│   │   ├── base.py               #     LLMAgent с retry и fallback
│   │   ├── polza_client.py       #     OpenAI-compatible через polza.ai
│   │   ├── orchestrator.py       #     Sequential run всех 9 ролей
│   │   ├── regime.py             #     #1 Regime Analyst
│   │   ├── analysts.py           #     #2 Technical, #4 Fundamental, #5 Pair
│   │   ├── news_analyst.py       #     #3 News Analyst (LLM)
│   │   ├── debate.py             #     #6 Bull / #7 Bear Researchers (LLM)
│   │   └── risk_pm.py            #     #8 Risk Officer / #9 Portfolio Manager (LLM)
│   │
│   ├── data/                     # ─── Источники данных + ML ───
│   │   ├── moex_iss.py           #     Кеш цен с TTL-fallback
│   │   ├── moex_candles.py       #     Async-batch свечи через aiomoex
│   │   ├── ml_signal.py          #     LightGBM + MacroContext overlay
│   │   ├── features.py           #     50 технических фичей
│   │   ├── indicators.py         #     RSI/MACD/ATR/momentum
│   │   ├── macro_context.py      #     logit-сдвиги из macro_priors.yaml
│   │   ├── volume_guard.py       #     Tracking оборота для ≥10М ₽
│   │   └── news/                 #     37 RSS-парсеров, 9 активных
│   │
│   ├── decision/
│   │   └── honey_money_provider.py  # Адаптер: orchestrator → ArenaGo orders
│   │
│   └── risk_extra/
│       ├── stop_loss.py          # StopLossManager с trailing + cooldown
│       └── crisis_gate.py        # Force-sell при IMOEX −2% за день
│
├── config/
│   └── macro_priors.yaml         # 24 макро-флага × 20 тикеров
├── models/
│   └── lightgbm_v1.txt           # Обученная модель (AUC test 0.560)
├── tests/                        # 157+ unit-тестов pytest
├── Dockerfile                    # python:3.12-slim + libgomp1
├── requirements.txt
├── .gitlab-ci.yml                # push_registry → deploy_helm → ops
├── .env.example                  # SANDBOX_API_KEY, POLZA_API_KEY, ...
└── README.md
```

---

## Команда

| Имя | Роль | Зона ответственности |
|---|---|---|
| **Валерия** | Team-lead, ML / AI-engineer | Архитектура, ML pipeline, LLM-граф, интеграция с ArenaGo |
| **Илья** | Frontend + Backend | Дашборд, мониторинг, инфраструктура |
| **Диана** | Designer + Developer | UI/UX, dark theme с красным акцентом |
| **Захар** | Trader + AI-developer | Параметры стратегии, бэктесты, риск-параметры |
| **Катерина** | Product / Marketing | Питч, презентация, документация |

---

## Этапы хакатона

| Этап | Даты | Что происходит | Где наш бот |
|---|---|---|---|
| **1** | 13–27 мая 2026, 15:00 | Разработка + ручная торговля, открытый лидерборд arenago.ru | Бот работает локально с ключом из ЛК команды |
| **2** | 28 мая 07:00 – 10 июня 15:00 МСК | Автономное тестирование, портфели сбрасываются, новые ключи | Organizers разворачивают Docker image из нашего registry автоматически |
| **3** | 11 – 30 июня | Экспертная оценка, проверка соблюдения правил | Решение зафиксировано, можем отдыхать |

Финальная оценка = `(70 × portfolio_value / leader_portfolio_value) + (0..30 баллов от эксперта)`.

---

## Лицензия

**Все права на код принадлежат автору. Использование запрещено** без письменного
разрешения правообладателя — см. файл [`LICENSE`](LICENSE). Код опубликован
исключительно для демонстрации и ознакомления; публикация не даёт прав на его
использование.

Внешние компоненты (модели, библиотеки) распространяются под своими открытыми
лицензиями — см. таблицу выше.
