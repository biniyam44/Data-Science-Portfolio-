"""Probe one local image through loopback HTTP, then remove its temporary container."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from uuid import uuid4

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_service import benchmark


def _docker(*arguments, env=None):
    return subprocess.run(["docker", *arguments], check=True, capture_output=True,
                          text=True, timeout=90, env=env).stdout.strip()


def validate(image, registry, payload_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    environment = {**os.environ, "FORECAST_API_TOKEN": token}
    name = "forecast-probe-" + uuid4().hex[:12]
    container = _docker(
        "run", "--detach", "--rm", "--pull", "never", "--name", name,
        "--publish", "127.0.0.1::8000", "--env", "FORECAST_API_TOKEN",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--pids-limit", "128", "--memory", "1g",
        "--mount", f"type=bind,src={Path(registry).resolve()},dst=/models,readonly",
        "--tmpfs", "/state:rw,noexec,nosuid,size=64m,uid=10001,gid=10001,mode=0700",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
        image, env=environment,
    )
    try:
        port = json.loads(_docker("inspect", "--format", '{{json (index .NetworkSettings.Ports "8000/tcp")}}', container))[0]["HostPort"]
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=url, timeout=30, trust_env=False, follow_redirects=False) as client:
            for _ in range(60):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("The container did not become healthy within 60 seconds.")
            unauthenticated = client.get("/ready").status_code
            assert unauthenticated == 401, "Readiness must require authentication."
            client.headers["Authorization"] = f"Bearer {token}"
            ready = client.get("/ready")
            ready.raise_for_status()
            payload = json.loads(Path(payload_path).read_text())
            forecast = client.post("/v1/forecast", json=payload)
            forecast.raise_for_status()
            response = forecast.json()
            assert len(response["forecasts"]) == 3
            analyst = client.post("/v1/ask", json={
                "question": "Which model has the lowest MAE for energy at six hours on test?",
            })
            analyst.raise_for_status()
            answer = analyst.json()
            assert answer["mode"] == "deterministic-evidence" and answer["citations"] and not answer["refused"]
            monitoring = client.get("/v1/monitoring")
            monitoring.raise_for_status()
            assert monitoring.json()["latest_drift"]["available"]
        runtime = json.loads(_docker("exec", container, "python", "-c",
            "import json,os,platform; print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),"
            "'system':platform.system(),'architecture':platform.machine(),'python':platform.python_version()}))"))
        assert runtime["uid"] == 10001 and runtime["gid"] == 10001
        security = json.loads(_docker("inspect", "--format", "{{json .HostConfig}}", container))
        assert security["ReadonlyRootfs"] is True and "ALL" in security["CapDrop"]
        mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", container))
        assert any(mount["Destination"] == "/models" and not mount["RW"] for mount in mounts)
        readonly = _docker("exec", container, "python", "-c",
            "import errno,pathlib\n"
            "for root in ['/models','/srv']:\n"
            " try:\n"
            "  with (pathlib.Path(root)/'container-validation-write-probe').open('x'): pass\n"
            " except OSError as exc:\n"
            "  assert exc.errno in (errno.EROFS,errno.EACCES), exc\n"
            " else: raise RuntimeError('Expected a read-only filesystem')\n"
            "print('read-only checks passed')")
        report = benchmark(url, payload, token)
        report["deployment"] = {"server_runtime": runtime, "server": "local Docker container",
                                "client_platform": report["platform"], "read_only_rootfs": True,
                                "model_mount_read_only": True, "worker_processes": 1}
        if report["failures"]:
            raise RuntimeError("At least one HTTP benchmark request failed.")
        artifacts = {
            "sample-response.json": response,
            "load-test.json": report,
            "http-probe.json": {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "image": image, "image_id": _docker("inspect", "--format", "{{.Image}}", container),
                "runtime": runtime, "health": "passed", "authenticated_readiness": ready.json(),
                "unauthenticated_readiness_status": unauthenticated,
                "root_filesystem": readonly, "model_mount_read_only": True,
                "dropped_capabilities": security["CapDrop"], "no_new_privileges": security["SecurityOpt"],
                "forecast_horizons": [item["horizon_hours"] for item in response["forecasts"]],
                "analyst_mode": answer["mode"], "analyst_citations": len(answer["citations"]),
                "drift_available": True, "paid_calls": 0,
                "scope": "One local container and loopback requests. No external deployment or paid model request.",
            },
        }
        for filename, value in artifacts.items():
            (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        _docker("stop", "--time", "5", container)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="biniyam-energy-service:local")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/energy"))
    parser.add_argument("--request", type=Path, default=Path("results/service/sample-request.json"))
    parser.add_argument("--output", type=Path, default=Path("results/service"))
    args = parser.parse_args()
    print(json.dumps(validate(args.image, args.registry, args.request, args.output), indent=2))


if __name__ == "__main__":
    main()
