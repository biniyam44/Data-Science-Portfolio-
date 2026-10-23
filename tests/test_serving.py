from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.serving import load_bundle, predict, promote, rollback, train_bundle


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("service")
    index = pd.date_range("2020-01-01", periods=1000, freq="h", name="timestamp")
    data = pd.DataFrame({"y": 1.0 + np.sin(np.arange(1000) * 2 * np.pi / 24) * 0.3
                         + np.arange(1000) * 0.0001}, index=index)
    path = root / "energy.csv"
    data.to_csv(path)
    metadata = train_bundle(path, root / "registry")
    return root, data, metadata


def test_offline_training_does_not_promote(trained):
    root, data, metadata = trained
    assert not (root / "registry" / "registry.json").exists()
    bundle = load_bundle(root / "registry", metadata["version"])
    assert bundle.metadata["training_cutoff"] == data.index[599].isoformat()
    assert bundle.metadata["calibration_cutoff"] == data.index[799].isoformat()
    for key, name in metadata["selected_models"].items():
        assert metadata["validation_mae"][key][name] <= metadata["validation_mae"][key]["naive"]


def test_prediction_roundtrip_and_frozen_estimator(trained, monkeypatch):
    root, data, metadata = trained
    bundle = load_bundle(root / "registry", metadata["version"])
    monkeypatch.setattr("forecasting.serving._fit", lambda *args: pytest.fail("Prediction attempted training"))
    response = predict(bundle, data.iloc[-169:])
    assert response["version"] == metadata["version"]
    assert [row["horizon_hours"] for row in response["forecasts"]] == [1, 6, 24]
    assert response == predict(load_bundle(root / "registry", metadata["version"]), data.iloc[-169:])
    for item in response["forecasts"]:
        assert 0 <= item["lower"] <= item["prediction"] <= item["upper"]
        assert pd.Timestamp(item["target_timestamp"]) == data.index[-1] + pd.Timedelta(hours=item["horizon_hours"])


@pytest.mark.parametrize("case", ["short", "reversed", "gap", "duplicate", "negative", "nan", "timezone", "minute", "volume", "boolean"])
def test_rejects_bad_prediction_history(trained, case):
    root, data, metadata = trained
    history = data.iloc[-200:].copy()
    if case == "short":
        history = history.iloc[-168:]
    elif case == "reversed":
        history = history.iloc[::-1]
    elif case == "gap":
        history = history.drop(history.index[10])
    elif case == "duplicate":
        history.index = history.index[:-1].append(history.index[-2:-1])
    elif case in {"negative", "nan"}:
        history.iloc[0, 0] = -1 if case == "negative" else np.nan
    elif case == "timezone":
        history.index = history.index.tz_localize("UTC")
    elif case == "minute":
        history.index += pd.Timedelta(minutes=1)
    elif case == "boolean":
        history["y"] = True
    else:
        history["volume"] = 1.0
    with pytest.raises(ValueError):
        predict(load_bundle(root / "registry", metadata["version"]), history)


def test_rejects_hindcast_before_calibration(trained):
    root, data, metadata = trained
    bundle = load_bundle(root / "registry", metadata["version"])
    with pytest.raises(ValueError, match="calibration cutoff"):
        predict(bundle, data.iloc[:799])
    assert predict(bundle, data.iloc[:800])["origin"] == data.index[799].isoformat()


@pytest.mark.parametrize("version", ["../models.joblib", "/tmp/model", "energy-nope", "", "energy-" + "a" * 24])
def test_version_names_are_not_paths(trained, version):
    root, _, _ = trained
    with pytest.raises(ValueError):
        load_bundle(root / "registry", version)


def test_tampered_metadata_fails_before_deserializing(trained, tmp_path, monkeypatch):
    import shutil

    root, _, metadata = trained
    registry = tmp_path / "registry"
    shutil.copytree(root / "registry", registry)
    path = registry / "bundles" / metadata["version"] / "metadata.json"
    path.write_text(path.read_text().replace('"interval_level":0.8', '"interval_level":0.99'))
    monkeypatch.setattr("forecasting.serving.joblib.load", lambda *_: pytest.fail("Unverified artifact loaded"))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(registry, metadata["version"])


def test_symlinked_model_is_rejected(trained, tmp_path):
    import shutil

    root, _, metadata = trained
    registry = tmp_path / "registry"
    shutil.copytree(root / "registry", registry)
    path = registry / "bundles" / metadata["version"] / "models.joblib"
    replacement = tmp_path / "model.joblib"
    path.rename(replacement)
    path.symlink_to(replacement)
    with pytest.raises(ValueError, match="regular files"):
        load_bundle(registry, metadata["version"])


def test_promotion_rollback_and_test_targets_excluded(trained, tmp_path):
    root, data, original = trained
    changed = data.copy()
    changed.iloc[800:, 0] = changed.iloc[800:, 0] * 100
    path = tmp_path / "changed.csv"
    changed.to_csv(path)
    metadata = train_bundle(path, root / "registry")
    assert metadata["version"] != original["version"]
    for field in ("selected_models", "validation_mae", "interval_half_widths", "reference_profile"):
        assert metadata[field] == original[field]
    before = load_bundle(root / "registry", original["version"])
    after = load_bundle(root / "registry", metadata["version"])
    assert predict(before, data)["forecasts"] == predict(after, data)["forecasts"]
    assert promote(root / "registry", original["version"])["active"] == original["version"]
    state = promote(root / "registry", metadata["version"])
    assert state["previous"] == original["version"]
    assert promote(root / "registry", metadata["version"]) == state
    assert rollback(root / "registry")["active"] == original["version"]
    assert load_bundle(root / "registry").version == original["version"]
    with pytest.raises(ValueError):
        load_bundle(root / "registry", "")


def test_empty_registry_cannot_rollback(tmp_path):
    with pytest.raises(ValueError, match="previous"):
        rollback(tmp_path)


def test_published_artifacts_are_readable_but_not_writable_by_service_uid(trained):
    root, _, metadata = trained
    registry = root / "registry"
    folder = registry / "bundles" / metadata["version"]
    assert folder.stat().st_mode & 0o777 == 0o755
    for name in ("metadata.json", "models.joblib", "manifest.json"):
        assert (folder / name).stat().st_mode & 0o777 == 0o644
    promote(registry, metadata["version"])
    assert (registry / "registry.json").stat().st_mode & 0o777 == 0o644
