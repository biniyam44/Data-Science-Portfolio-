"""Measure a bounded, authenticated loopback HTTP workload, never a public target."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time
from urllib.parse import urlparse

import httpx
import numpy as np


def benchmark(url, payload, token, requests=20, concurrency=4):
    if not isinstance(url, str) or any(ord(character) < 33 for character in url):
        raise ValueError("Use a valid loopback URL without whitespace.")
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("This command only benchmarks a loopback HTTP server.")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("Use a valid TCP port.")
    if (type(requests) is not int or type(concurrency) is not int
            or not 1 <= requests <= 30 or not 1 <= concurrency <= 8
            or not isinstance(token, str) or len(token) < 32 or any(character.isspace() for character in token)):
        raise ValueError("Use 1..30 requests, 1..8 workers, and a configured token.")
    if not isinstance(payload, dict) or len(json.dumps(payload, allow_nan=False).encode()) > 128000:
        raise ValueError("The request must be a finite JSON object below 128 KB.")
    if parsed.hostname == "localhost":
        # Do not let a modified hosts file turn a loopback-only test into a remote workload.
        url = f"http://127.0.0.1:{parsed.port or 80}"
    with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"},
                      timeout=30, follow_redirects=False, trust_env=False) as client:
        warm = client.post("/v1/forecast", json=payload)
        warm.raise_for_status()
        warm_result = warm.json()
        if not isinstance(warm_result, dict) or not re.fullmatch(r"energy-[0-9a-f]{24}", str(warm_result.get("version", ""))):
            raise ValueError("Warmup did not return a valid energy model version.")
        version = warm_result["version"]

        def request(_):
            start = time.perf_counter()
            status = 0
            try:
                response = client.post("/v1/forecast", json=payload)
                status = response.status_code
                result = response.json() if status == 200 else None
                valid = status == 200 and isinstance(result, dict) and result.get("version") == version
                return (time.perf_counter() - start) * 1000, valid, status
            except (httpx.HTTPError, ValueError):
                return (time.perf_counter() - start) * 1000, False, status

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            samples = list(pool.map(request, range(requests)))
        elapsed = max(time.perf_counter() - start, 1e-9)
    latencies = [latency for latency, valid, _ in samples if valid]
    return {
        "kind": "local HTTP load test", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
        "model_version": version, "requests": requests, "concurrency": concurrency,
        "warmup_requests_excluded": 1, "successes": len(latencies),
        "failures": requests - len(latencies), "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(latencies) / elapsed,
        "latency_ms": {"p50": float(np.percentile(latencies, 50)) if latencies else None,
                       "p95": float(np.percentile(latencies, 95)) if latencies else None},
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "status_codes": [status for _, _, status in samples],
        "limits": "One warmed process on one local machine; not internet latency or a production capacity claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--request", type=Path, default=Path("results/service/sample-request.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/load-test.json"))
    args = parser.parse_args()
    report = benchmark(args.url, json.loads(args.request.read_text()), os.environ.get("FORECAST_API_TOKEN", ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
