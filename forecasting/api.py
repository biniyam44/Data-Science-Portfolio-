"""Authenticated single-process API for saved energy models and research evidence."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hmac import compare_digest
import os
from pathlib import Path
import threading
import time
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[1]
HOURLY_FORMAT = "%Y-%m-%dT%H:%M:%S"
MAX_ENERGY_KWH = 1_000_000
MIN_SOURCE_TIMESTAMP = pd.Timestamp.min.ceil("s").to_pydatetime()
MAX_SOURCE_TIMESTAMP = pd.Timestamp.max.floor("s").to_pydatetime()


@dataclass(frozen=True)
class Settings:
    registry: Path = field(default_factory=lambda: Path(os.getenv("FORECAST_REGISTRY", ROOT / "artifacts/energy")))
    monitoring: Path = field(default_factory=lambda: Path(os.getenv("FORECAST_MONITOR_DB", ROOT / "var/monitor.db")))
    repo_root: Path = ROOT
    api_token: str = field(default_factory=lambda: os.getenv("FORECAST_API_TOKEN", ""))
    requests_per_minute: int = 60
    max_body_bytes: int = 256_000
    llm_enabled: bool = field(default_factory=lambda: os.getenv("FORECAST_ENABLE_LLM") == "1")

    def __post_init__(self):
        if self.requests_per_minute < 1 or self.max_body_bytes < 1024:
            raise ValueError("Positive request limits are required.")
        if self.api_token and len(self.api_token) < 32:
            raise ValueError("FORECAST_API_TOKEN must contain at least 32 characters.")


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def source_timestamp(value: str) -> str:
    try:
        stamp = datetime.strptime(value, HOURLY_FORMAT)
    except (ValueError, TypeError) as error:
        raise ValueError("Use a source-clock timestamp such as 2016-05-01T12:00:00, without a timezone offset.") from error
    if stamp.strftime(HOURLY_FORMAT) != value:
        raise ValueError("Use the exact YYYY-MM-DDTHH:MM:SS source-clock format.")
    if not MIN_SOURCE_TIMESTAMP <= stamp <= MAX_SOURCE_TIMESTAMP:
        raise ValueError("The source timestamp is outside the supported calendar range.")
    return value


def hourly_timestamp(value: str) -> str:
    source_timestamp(value)
    stamp = datetime.strptime(value, HOURLY_FORMAT)
    if stamp.minute or stamp.second:
        raise ValueError("Every reading must start on an exact hour.")
    return value


class Reading(StrictBody):
    timestamp: Annotated[str, Field(strict=True, max_length=19)]
    y: Annotated[float, Field(strict=True, ge=0, le=MAX_ENERGY_KWH)]
    _hour = field_validator("timestamp")(hourly_timestamp)


class ForecastRequest(StrictBody):
    unit: Literal["kWh per hour"]
    history: Annotated[list[Reading], Field(min_length=169, max_length=1680)]
    request_id: Annotated[str | None, Field(pattern=r"^[a-f0-9]{32}$", strict=True)] = None

    @model_validator(mode="after")
    def contiguous(self):
        stamps = [datetime.strptime(row.timestamp, HOURLY_FORMAT) for row in self.history]
        if any(right - left != timedelta(hours=1) for left, right in zip(stamps, stamps[1:])):
            raise ValueError("History must be ordered, unique and hourly, without gaps.")
        if stamps[-1] > MAX_SOURCE_TIMESTAMP - timedelta(hours=25):
            raise ValueError("The forecast origin must leave room for every target hour and its observation.")
        return self


class OutcomeRequest(StrictBody):
    request_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    target_timestamp: Annotated[str, Field(strict=True, max_length=19)]
    actual: Annotated[float, Field(strict=True, ge=0, le=MAX_ENERGY_KWH)]
    observed_at: Annotated[str, Field(strict=True, max_length=19)]
    unit: Literal["kWh per hour"] = "kWh per hour"
    _hour = field_validator("target_timestamp")(hourly_timestamp)
    _observed = field_validator("observed_at")(source_timestamp)


class AnalystRequest(StrictBody):
    question: Annotated[str, Field(strict=True, min_length=5, max_length=1500)]
    use_llm: Annotated[bool, Field(strict=True)] = False


class BodyLimit:
    """Bound even chunked requests before JSON parsing or authentication work."""
    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > self.maximum:
                return await JSONResponse({"detail": "Request body is too large."}, status_code=413)(scope, receive, send)
            body.extend(chunk)
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class AccessLimit:
    """Authenticate before parsing client bodies; bound authenticated work per process."""

    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.arrivals = deque()
        self.lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health":
            return await self.app(scope, receive, send)
        token = self.settings.api_token
        if not token:
            return await JSONResponse({"detail": "Service access is not configured."}, 503)(scope, receive, send)
        authorization = [value for name, value in scope.get("headers", []) if name.lower() == b"authorization"]
        pieces = authorization[0].split(b" ", 1) if len(authorization) == 1 else []
        if len(pieces) != 2 or pieces[0].lower() != b"bearer" or not compare_digest(pieces[1], token.encode()):
            return await JSONResponse({"detail": "A valid bearer token is required."}, 401,
                                      headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
        now = time.monotonic()
        with self.lock:
            while self.arrivals and self.arrivals[0] <= now - 60:
                self.arrivals.popleft()
            limited = len(self.arrivals) >= self.settings.requests_per_minute
            if not limited:
                self.arrivals.append(now)
        if limited:
            return await JSONResponse({"detail": "Request limit reached."}, 429,
                                      headers={"Retry-After": "60"})(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Energy Forecasting API", version="0.2.0", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit, maximum=settings.max_body_bytes)
    app.add_middleware(AccessLimit, settings=settings)
    bearer = HTTPBearer(auto_error=False)
    load_lock = threading.Lock()
    app.state.bundle = None
    app.state.evidence = None
    app.state.monitor = None
    app.state.settings = settings

    def monitor():
        if app.state.monitor is None:
            from .monitoring import Monitor
            with load_lock:
                if app.state.monitor is None:
                    app.state.monitor = Monitor(settings.monitoring)
        return app.state.monitor

    def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if not settings.api_token:
            raise HTTPException(503, "Service access is not configured.")
        if credentials is None or not compare_digest(credentials.credentials.encode(), settings.api_token.encode()):
            raise HTTPException(401, "A valid bearer token is required.", headers={"WWW-Authenticate": "Bearer"})

    def bundle():
        if app.state.bundle is None:
            from .serving import load_bundle
            with load_lock:
                if app.state.bundle is None:
                    try:
                        app.state.bundle = load_bundle(settings.registry)
                    except (ValueError, OSError, KeyError) as error:
                        raise HTTPException(503, "No valid promoted model is available.") from error
        return app.state.bundle

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        if request.url.path == "/v1/forecast":
            missing = any(item["type"] in {"missing", "too_short"} for item in error.errors())
            monitor().record_input_failure("missing_input" if missing else "invalid_input")
        # Do not echo client payloads or token-bearing request context into errors.
        issues = [{"location": list(item["loc"]), "message": item["msg"], "type": item["type"]} for item in error.errors()]
        return JSONResponse({"detail": issues}, status_code=422)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "energy-forecasting", "authentication_configured": bool(settings.api_token)}

    @app.get("/ready", dependencies=[Depends(authorize)])
    def ready():
        model = bundle()
        return {"status": "ready", "model_version": model.version}

    @app.get("/openapi.json", dependencies=[Depends(authorize)])
    def openapi():
        return app.openapi()

    @app.get("/v1/model", dependencies=[Depends(authorize)])
    def model_info():
        model = bundle()
        return {"version": model.version, **model.metadata}

    @app.post("/v1/forecast", dependencies=[Depends(authorize)])
    def forecast(payload: ForecastRequest):
        from .serving import predict
        started = time.perf_counter()
        history = pd.DataFrame({"y": [row.y for row in payload.history]}, index=pd.DatetimeIndex([row.timestamp for row in payload.history]))
        model = bundle()
        try:
            result = predict(model, history)
        except ValueError as error:
            monitor().record_input_failure()
            raise HTTPException(422, str(error)) from error
        elapsed = (time.perf_counter() - started) * 1000
        request_id = payload.request_id or uuid4().hex
        try:
            recorded = monitor().record_forecast(request_id, result, elapsed, history, model.metadata["reference_profile"])
        except ValueError as error:
            monitor().record_input_failure("conflicting_request")
            raise HTTPException(409, "Request ID conflicts with a previously recorded forecast.") from error
        return {**result, "request_id": request_id, "latency_ms": elapsed, "created": recorded["created"]}

    @app.post("/v1/outcomes", dependencies=[Depends(authorize)])
    def outcome(payload: OutcomeRequest):
        try:
            outcome_result = monitor().record_outcome(**payload.model_dump())
        except KeyError as error:
            raise HTTPException(404, "Forecast request or target was not found.") from error
        except ValueError as error:
            if str(error).startswith(("No forecast exists", "The outcome target is not")):
                raise HTTPException(404, "Forecast request or target was not found.") from error
            if "conflicts with" in str(error):
                raise HTTPException(409, str(error)) from error
            raise HTTPException(422, str(error)) from error
        return {"status": "recorded", "result": outcome_result}

    @app.get("/v1/monitoring", dependencies=[Depends(authorize)])
    def monitoring(as_of: str | None = None):
        try:
            return monitor().report(as_of=source_timestamp(as_of) if as_of is not None else None)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/ask", dependencies=[Depends(authorize)])
    def ask(payload: AnalystRequest):
        from .analyst import EvidenceStore
        if payload.use_llm and not settings.llm_enabled:
            raise HTTPException(403, "Paid model access is disabled. Use deterministic evidence mode.")
        if app.state.evidence is None:
            with load_lock:
                if app.state.evidence is None:
                    try:
                        app.state.evidence = EvidenceStore(settings.repo_root)
                    except (ValueError, OSError, KeyError) as error:
                        raise HTTPException(503, "The saved research evidence failed validation.") from error
        planner = None
        if payload.use_llm:
            from .analyst import CompatibleToolPlanner
            try:
                planner = CompatibleToolPlanner.from_env(allow_paid=True)
            except (ValueError, KeyError) as error:
                raise HTTPException(503, "Model access or spending limits are not configured.") from error
        try:
            return app.state.evidence.answer(payload.question, planner=planner)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(502, "The model request could not be completed. No automatic retry was made.") from error

    return app


app = create_app()
