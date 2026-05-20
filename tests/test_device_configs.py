"""
Device descriptor config validation — Task 1.4 acceptance check.

Discovers all YAML files in configs/devices/ and validates each one through
the DeviceDescriptor Pydantic model. Fails if any file is missing a required
field, uses an unknown enum value, or violates a cross-field constraint
(e.g. preferred_runtime not in supported_runtimes).

Also checks that every device in tests/fixtures/ has a matching config file
in configs/devices/, keeping the two in sync.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent

from schemas import DeviceDescriptor

DEVICES_DIR = ROOT / "configs" / "devices"
FIXTURES_DIR = ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _device_yaml_paths() -> list[Path]:
    return sorted(DEVICES_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Sync check: configs/devices/ and tests/fixtures/ must agree on device IDs
# ---------------------------------------------------------------------------

def test_device_configs_match_fixtures() -> None:
    """Every device YAML must have a matching fixture and vice versa.

    Matching rule: configs/devices/foo.yaml ↔ tests/fixtures/device_descriptor_foo.json.
    A mismatch means the two were edited independently and have drifted.
    """
    config_stems = {p.stem for p in _device_yaml_paths()}
    fixture_stems = {
        p.stem.removeprefix("device_descriptor_")
        for p in FIXTURES_DIR.glob("device_descriptor_*.json")
    }

    only_in_configs = config_stems - fixture_stems
    only_in_fixtures = fixture_stems - config_stems

    assert not only_in_configs, (
        "Device YAML configs with no matching fixture in tests/fixtures/:\n  "
        + "\n  ".join(sorted(only_in_configs))
    )
    assert not only_in_fixtures, (
        "Device fixtures with no matching YAML config in configs/devices/:\n  "
        + "\n  ".join(sorted(only_in_fixtures))
    )


# ---------------------------------------------------------------------------
# Per-file parametrised validation
# ---------------------------------------------------------------------------

_YAML_PATHS = _device_yaml_paths()
_YAML_IDS = [p.stem for p in _YAML_PATHS]


@pytest.mark.parametrize("yaml_path", _YAML_PATHS, ids=_YAML_IDS)
def test_device_config_valid(yaml_path: Path) -> None:
    """Device YAML loads and validates through the DeviceDescriptor Pydantic model."""
    raw = yaml.safe_load(yaml_path.read_text())
    try:
        device = DeviceDescriptor.model_validate(raw)
    except Exception as exc:
        pytest.fail(
            f"DeviceDescriptor validation failed for {yaml_path.name}:\n{exc}"
        )

    # Spot-checks that are easy to misconfigure in YAML
    assert device.device_id == yaml_path.stem, (
        f"device_id '{device.device_id}' does not match filename '{yaml_path.stem}'. "
        "Keep them in sync so the Deployment Dispatcher can load a descriptor by device_id."
    )
    assert len(device.quirks) >= 1, (
        f"{yaml_path.name}: quirks list is empty. "
        "Every device has at least one operational note worth recording."
    )
