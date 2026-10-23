"""Train frozen energy models offline and load verified, server-owned bundles."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .data import load_data, validate_data
from .evaluation import calibration_width, make_origin_plan
from .features import feature_matrix, supervised_rows

HORIZONS = (1, 6, 24)
UNIT = "kWh per hour"
MIN_HISTORY = 169
MODEL_NAMES = ("naive", "seasonal_naive", "boosted")
MODEL_PARAMETERS = {
    "max_iter": 80, "max_leaf_nodes": 15,
    "early_stopping": False, "random_state": 42,
}
VERSION_RE = re.compile(r"energy-[0-9a-f]{24}\Z")


@dataclass(frozen=True)
class Bundle:
    version: str
    metadata: dict
    models: dict


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _dependencies() -> dict:
    return {name: package_version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}


def _implementation_hash() -> str:
    root = Path(__file__).parent
    digest = sha256()
    for name in ("serving.py", "features.py", "data.py", "evaluation.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _history(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_monotonic_increasing:
        raise ValueError("Readings must be supplied in increasing timestamp order.")
    if set(data.columns) != {"y"}:
        raise ValueError("Energy history accepts only the y measurement column.")
    if any(isinstance(value, (bool, np.bool_)) for value in data["y"]):
        raise ValueError("Boolean values are not energy measurements.")
    if data.index.tz is not None:
        raise ValueError("Use the documented timezone-naive source clock.")
    if any(t.minute or t.second or t.microsecond or t.nanosecond for t in data.index):
        raise ValueError("Readings must be aligned to the start of each hour.")
    validated = validate_data(data, "h")
    if len(validated) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} completed hourly readings are required.")
    return validated


def _fit(data: pd.DataFrame, horizon: int):
    x, y, _ = supervised_rows(data, horizon, "h")
    estimator = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
    with threadpool_limits(limits=1):
        estimator.fit(x, y)
    return estimator


def _estimate(name: str, model, history: pd.DataFrame, horizon: int) -> float:
    if name == "naive":
        result = history["y"].iloc[-1]
    elif name == "seasonal_naive":
        result = history["y"].iloc[-24 + ((horizon - 1) % 24)]
    elif name == "boosted":
        x = feature_matrix(history, horizon, "h").iloc[[-1]]
        with threadpool_limits(limits=1):
            result = history["y"].iloc[-1] + model.predict(x)[0]
    else:
        raise ValueError("Unknown model in bundle.")
    if not np.isfinite(result):
        raise ValueError("The model returned a nonfinite prediction.")
    return max(0.0, float(result))


def _phase_errors(data, origins, horizon, name, estimator):
    predictions = np.asarray([
        _estimate(name, estimator, data.iloc[:origin + 1], horizon) for origin in origins
    ])
    actual = data["y"].iloc[np.asarray(origins) + horizon].to_numpy()
    return np.abs(actual - predictions)


def _profile(values: np.ndarray) -> dict:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if minimum == maximum:
        maximum = minimum + 1.0
    edges = np.linspace(minimum, maximum, 11)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "minimum": minimum, "maximum": float(np.max(values)),
        "bin_edges": edges.tolist(), "bin_counts": counts.tolist(),
        "rows": len(values),
    }


def train_bundle(data_path: str | Path, registry_dir: str | Path) -> dict:
    """Fit, validate and calibrate without reading the final test targets.

    The final 20 percent is reserved for a separate evaluation. This historical
    dataset was previously explored, so this is not a new external holdout.
    """
    data = _history(load_data(data_path, "energy"))
    plan, splits = make_origin_plan(data, "energy")
    train_end = splits["training"]["end_index_exclusive"]
    validation_end = splits["validation"]["end_index_exclusive"]
    calibration_end = splits["calibration"]["end_index_exclusive"]
    selected, models, scores, widths = {}, {}, {}, {}
    for horizon in HORIZONS:
        challenger = _fit(data.iloc[:train_end], horizon)
        errors = {
            name: _phase_errors(data, plan["validation"], horizon, name, challenger)
            for name in MODEL_NAMES
        }
        metrics = {name: float(values.mean()) for name, values in errors.items()}
        winner = min(MODEL_NAMES, key=lambda name: (metrics[name], name))
        key = str(horizon)
        selected[key] = winner
        scores[key] = metrics
        models[key] = _fit(data.iloc[:validation_end], horizon) if winner == "boosted" else None
        residuals = _phase_errors(data, plan["calibration"], horizon, winner, models[key])
        widths[key] = calibration_width(residuals, 0.8)
    metadata = {
        "schema_version": 1, "dataset": "energy", "unit": UNIT,
        "frequency": "h", "clock": "timezone-naive source-local timestamps",
        "horizons": list(HORIZONS), "minimum_history": MIN_HISTORY,
        "feature_columns": feature_matrix(data.iloc[:train_end], 1, "h").columns.tolist(),
        "feature_implementation_sha256": _implementation_hash(),
        "training_cutoff": data.index[validation_end - 1].isoformat(),
        "selection_training_cutoff": data.index[train_end - 1].isoformat(),
        "calibration_cutoff": data.index[calibration_end - 1].isoformat(),
        "input_sha256": sha256(Path(data_path).read_bytes()).hexdigest(),
        "dependencies": _dependencies(), "parameters": MODEL_PARAMETERS,
        "selected_models": selected, "validation_mae": scores,
        "interval_level": 0.8, "interval_half_widths": widths,
        "calibration_origins": len(plan["calibration"]), "splits": splits,
        "reference_profile": _profile(data["y"].iloc[:validation_end].to_numpy()),
        "promotion_policy": "Per horizon, use the minimum validation MAE among naive, seasonal_naive and boosted; ties use model name. Test outcomes do not influence promotion.",
        "evaluation_status": "Final test targets have not been scored by this training command. The source dataset was previously explored.",
        "interval_caveat": "Empirical historical residual intervals, not a coverage guarantee under dependence or drift.",
    }
    root = _registry_root(registry_dir)
    bundles = root / "bundles"
    bundles.mkdir(exist_ok=True)
    if bundles.is_symlink():
        raise ValueError("Registry bundle directory must not be a symlink.")
    stage = Path(tempfile.mkdtemp(prefix=".training-", dir=bundles))
    try:
        joblib.dump(models, stage / "models.joblib", compress=3)
        (stage / "metadata.json").write_bytes(_json_bytes(metadata))
        files = {name: sha256((stage / name).read_bytes()).hexdigest()
                 for name in ("metadata.json", "models.joblib")}
        manifest = {"schema_version": 1, "files": files}
        version = "energy-" + sha256(_json_bytes(manifest)).hexdigest()[:24]
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        # Serving runs as another unprivileged UID. These artifacts contain no
        # credentials; published files are readable but never writable by it.
        for name in ("models.joblib", "metadata.json", "manifest.json"):
            (stage / name).chmod(0o644)
        stage.chmod(0o755)
        destination = bundles / version
        with _registry_lock(root):
            if destination.exists():
                load_bundle(root, version)
            else:
                stage.rename(destination)
        return {"version": version, **metadata}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _registry_root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if root.is_symlink():
        raise ValueError("Registry directory must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _version(value: str) -> str:
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
        raise ValueError("Version must be an energy bundle identifier, not a path.")
    return value


def _read_state(root: Path) -> dict:
    path = root / "registry.json"
    if not path.exists():
        return {"schema_version": 1, "active": None, "previous": None, "events": []}
    if path.is_symlink() or not path.is_file():
        raise ValueError("Registry state must be a regular file.")
    state = json.loads(path.read_text())
    if (state.get("schema_version") != 1 or not isinstance(state.get("events"), list)
            or set(state) != {"schema_version", "active", "previous", "events"}):
        raise ValueError("Invalid registry state.")
    for key in ("active", "previous"):
        if state[key] is not None:
            _version(state[key])
    return state


@contextmanager
def _registry_lock(root: Path):
    import fcntl

    fd = os.open(root / ".registry.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_state(root: Path, state: dict):
    fd, name = tempfile.mkstemp(prefix=".registry-", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(state))
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, root / "registry.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_bundle(registry_dir: str | Path, version: str | None = None) -> Bundle:
    """Only load locally trained artifacts from a trusted, server-owned registry.

    Hashes detect corruption, not malicious replacement by a registry owner.
    Joblib deserialization executes Python and must never accept uploaded files.
    """
    root = _registry_root(registry_dir)
    version = _version(_read_state(root)["active"] if version is None else version)
    folder = root / "bundles" / version
    if folder.is_symlink() or folder.parent.is_symlink() or not folder.is_dir():
        raise ValueError("Unknown or unsafe model version.")
    for name in ("manifest.json", "metadata.json", "models.joblib"):
        if (folder / name).is_symlink() or not (folder / name).is_file():
            raise ValueError("Bundle files must be regular files.")
    manifest = json.loads((folder / "manifest.json").read_text())
    if (set(manifest) != {"schema_version", "files"} or manifest["schema_version"] != 1
            or set(manifest["files"]) != {"metadata.json", "models.joblib"}):
        raise ValueError("Invalid bundle manifest.")
    if "energy-" + sha256(_json_bytes(manifest)).hexdigest()[:24] != version:
        raise ValueError("Bundle manifest does not match its version.")
    for name, expected in manifest["files"].items():
        if sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    metadata = json.loads((folder / "metadata.json").read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("unit") != UNIT
            or metadata.get("horizons") != list(HORIZONS)
            or metadata.get("dependencies") != _dependencies()
            or metadata.get("feature_implementation_sha256") != _implementation_hash()):
        raise ValueError("Bundle is incompatible with this feature implementation or environment.")
    for horizon in HORIZONS:
        key = str(horizon)
        scores = metadata["validation_mae"][key]
        if set(scores) != set(MODEL_NAMES) or any(not np.isfinite(v) or v < 0 for v in scores.values()):
            raise ValueError("Invalid validation metrics.")
        if metadata["selected_models"][key] != min(MODEL_NAMES, key=lambda name: (scores[name], name)):
            raise ValueError("Bundle violates the validation promotion rule.")
        width = metadata["interval_half_widths"][key]
        if not np.isfinite(width) or width < 0:
            raise ValueError("Invalid interval calibration.")
    models = joblib.load(folder / "models.joblib")
    if not isinstance(models, dict) or set(models) != {str(h) for h in HORIZONS}:
        raise ValueError("Unexpected estimator bundle.")
    for key, model in models.items():
        if metadata["selected_models"][key] == "boosted":
            if not isinstance(model, HistGradientBoostingRegressor):
                raise ValueError("Unexpected estimator type.")
            if list(model.feature_names_in_) != metadata["feature_columns"]:
                raise ValueError("Estimator feature schema mismatch.")
        elif model is not None:
            raise ValueError("Baseline bundles must not contain an estimator.")
    return Bundle(version, metadata, models)


def predict(bundle: Bundle, history: pd.DataFrame) -> dict:
    history = _history(history)
    origin = history.index[-1]
    metadata = bundle.metadata
    if origin < pd.Timestamp(metadata["calibration_cutoff"]):
        raise ValueError("Forecast origin precedes the bundle's calibration cutoff.")
    forecasts = []
    for horizon in HORIZONS:
        key = str(horizon)
        if list(feature_matrix(history, horizon, "h").columns) != metadata["feature_columns"]:
            raise ValueError("Input feature schema does not match the bundle.")
        estimate = _estimate(metadata["selected_models"][key], bundle.models[key], history, horizon)
        width = metadata["interval_half_widths"][key]
        forecasts.append({
            "horizon_hours": horizon, "target_timestamp": (origin + pd.Timedelta(hours=horizon)).isoformat(),
            "prediction": estimate, "lower": max(0.0, estimate - width), "upper": estimate + width,
            "model": metadata["selected_models"][key],
        })
    return {
        "version": bundle.version, "origin": origin.isoformat(),
        "training_cutoff": metadata["training_cutoff"], "calibration_cutoff": metadata["calibration_cutoff"],
        "unit": UNIT, "interval_level": metadata["interval_level"], "forecasts": forecasts,
        "interval_caveat": metadata["interval_caveat"],
    }


def promote(registry_dir: str | Path, version: str) -> dict:
    root = _registry_root(registry_dir)
    load_bundle(root, version)
    with _registry_lock(root):
        state = _read_state(root)
        if state["active"] == version:
            return state
        state["previous"], state["active"] = state["active"], version
        state["events"].append({"action": "promote", "version": version})
        _write_state(root, state)
    return state


def rollback(registry_dir: str | Path) -> dict:
    root = _registry_root(registry_dir)
    with _registry_lock(root):
        state = _read_state(root)
        if state["previous"] is None:
            raise ValueError("There is no previous promoted version to restore.")
        load_bundle(root, state["previous"])
        state["active"], state["previous"] = state["previous"], state["active"]
        state["events"].append({"action": "rollback", "version": state["active"]})
        _write_state(root, state)
    return state
