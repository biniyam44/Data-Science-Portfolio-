FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    FORECAST_REGISTRY=/models FORECAST_MONITOR_DB=/state/monitor.db
WORKDIR /srv
COPY requirements-service.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 3 -r requirements-service.lock \
    && groupadd --gid 10001 forecast \
    && useradd --uid 10001 --gid forecast --no-create-home forecast \
    && mkdir /models /state && chown forecast:forecast /models /state
COPY forecasting/ ./forecasting/
COPY scripts/ ./scripts/
COPY data/dc.csv data/appliances_hourly.csv ./data/
COPY results/crypto/ ./results/crypto/
COPY results/energy/ ./results/energy/
COPY docs/methodology.md ./docs/methodology.md
RUN chmod -R a+rX /srv
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "forecasting.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
