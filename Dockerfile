FROM python:3.12-slim

# Системные библиотеки для LightGBM (libgomp1 — OpenMP runtime)
# и для lxml (используется в news-парсерах, опционально)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# /data — постоянное хранилище (state, audit, stop_loss, turnover).
# Organizers монтируют сюда диск; создаём папку заранее на случай read-only mount или non-root user.
RUN mkdir -p /data && chmod 777 /data

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Moscow \
    DATA_DIR=/data

EXPOSE 8080

# Healthcheck для мониторинга organizers (read-only endpoint).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8080/healthz || exit 1

CMD ["python", "-m", "src.main"]
