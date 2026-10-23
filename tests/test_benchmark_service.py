import hashlib
import json

import httpx
import pytest

from scripts import benchmark_service as module


VERSION = "energy-" + "a" * 24
TOKEN = "test-token-" + "x" * 32
PAYLOAD = {"unit": "kWh per hour", "history": []}


def client_mock(monkeypatch, handler):
    original = httpx.Client
    options = []
    def factory(**kwargs):
        options.append(kwargs.copy())
        return original(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module.httpx, "Client", factory)
    return options


@pytest.mark.parametrize("url", ["https://127.0.0.1", "http://example.com", "http://127.0.0.2", "http://127.0.0.1.evil.example", "http://user@localhost", "http://localhost/path", "http://localhost?x=y", "http://localhost#fragment", "http://localhost:0", "http://localhost:99999", "http://local\nhost", None])
def test_remote_or_ambiguous_urls_never_reach_network(monkeypatch, url):
    monkeypatch.setattr(module.httpx, "Client", lambda **_: pytest.fail("Unexpected network request"))
    with pytest.raises(ValueError):
        module.benchmark(url, PAYLOAD, TOKEN)


@pytest.mark.parametrize("kwargs", [{"requests": 0}, {"requests": 31}, {"requests": True}, {"requests": 1.5}, {"concurrency": 0}, {"concurrency": 9}, {"concurrency": True}, {"token": "short"}, {"token": None}, {"token": "x" * 32 + "\n"}, {"payload": []}, {"payload": {"bad": float("nan")}}, {"payload": {"large": "x" * 128001}}])
def test_invalid_workload_refused_before_network(monkeypatch, kwargs):
    monkeypatch.setattr(module.httpx, "Client", lambda **_: pytest.fail("Unexpected network request"))
    values = {"url": "http://localhost:8000", "payload": PAYLOAD, "token": TOKEN, **kwargs}
    with pytest.raises(ValueError):
        module.benchmark(**values)


def test_successes_exclude_warmup_and_bind_request(monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        assert json.loads(request.content) == PAYLOAD
        return httpx.Response(200, json={"version": VERSION})
    options = client_mock(monkeypatch, handler)
    result = module.benchmark("http://localhost:8000", PAYLOAD, TOKEN, requests=3, concurrency=2)
    assert len(calls) == 4 and result["successes"] == 3 and result["failures"] == 0
    assert result["warmup_requests_excluded"] == 1 and result["status_codes"] == [200, 200, 200]
    assert result["request_sha256"] == hashlib.sha256(json.dumps(PAYLOAD, sort_keys=True).encode()).hexdigest()
    assert result["successful_requests_per_second"] == pytest.approx(3 / result["elapsed_seconds"])
    assert 0 <= result["latency_ms"]["p50"] <= result["latency_ms"]["p95"]
    assert options[0]["trust_env"] is False and options[0]["follow_redirects"] is False
    assert TOKEN not in json.dumps(result)


@pytest.mark.parametrize("failure,code", [("redirect", 302), ("unauthorized", 401), ("rate_limit", 429), ("different_version", 200), ("invalid_json", 200), ("non_object", 200), ("timeout", 0)])
def test_measured_failures_keep_status_and_do_not_become_success(monkeypatch, failure, code):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(200, json={"version": VERSION})
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        if failure == "invalid_json":
            return httpx.Response(200, text="bad")
        if failure == "non_object":
            return httpx.Response(200, json=[])
        if failure == "different_version":
            return httpx.Response(200, json={"version": "energy-" + "b" * 24})
        return httpx.Response(code, json={}, headers={"Location": "http://example.com"})
    client_mock(monkeypatch, handler)
    result = module.benchmark("http://127.0.0.1:8000", PAYLOAD, TOKEN, requests=2, concurrency=1)
    assert count == 3 and result["successes"] == 0 and result["failures"] == 2
    assert result["status_codes"] == [code, code] and result["latency_ms"] == {"p50": None, "p95": None}
    assert result["successful_requests_per_second"] == 0


@pytest.mark.parametrize("payload", [{}, {"version": ""}, {"version": "made-up"}, [], None])
def test_invalid_warmup_stops_before_measured_requests(monkeypatch, payload):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)
    client_mock(monkeypatch, handler)
    with pytest.raises(ValueError):
        module.benchmark("http://[::1]:8000", PAYLOAD, TOKEN)
    assert len(calls) == 1


def test_cli_writes_failed_measurement_and_exits_nonzero(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    output = tmp_path / "report.json"
    request.write_text(json.dumps(PAYLOAD))
    monkeypatch.setattr(module, "benchmark", lambda *args: {"failures": 1, "successes": 0})
    monkeypatch.setattr("sys.argv", ["benchmark_service.py", "--request", str(request), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1 and json.loads(output.read_text())["failures"] == 1
