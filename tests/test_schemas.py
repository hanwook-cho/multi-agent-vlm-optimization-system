"""
Schema and fixture consistency tests — Task 1.3 acceptance check.

Discovers all *.schema.json files in schemas/ and all fixture files in
tests/fixtures/. For each fixture:
  - Validates it against the matching JSON Schema (jsonschema).
  - Loads it through the matching Pydantic model and round-trips it back
    to JSON, then re-validates the round-tripped form.

Also fails loudly if any fixture lacks a matching schema, any schema lacks
a fixture, or SCHEMA_MODEL_MAP is missing an entry for a discovered schema.

Naming convention: fixture filename must start with '{schema_name}_'
where schema_name is the *.schema.json stem without '.schema'.
Example: 'device_descriptor_iphone_16_pro.json' matches
         'device_descriptor.schema.json'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

ROOT = Path(__file__).parent.parent

from schemas import (
    AgentDecision,
    DeviceDescriptor,
    ExperimentConfig,
    HypothesisRecord,
    MetricsReport,
    StudentSpec,
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

SCHEMAS_DIR = ROOT / "schemas"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Schema name → Pydantic model
# Update this map whenever a new schema is added to schemas/.
# test_schema_model_map_covers_all_schemas will fail if a schema is missing.
# ---------------------------------------------------------------------------

SCHEMA_MODEL_MAP: dict[str, Any] = {
    "device_descriptor": DeviceDescriptor,
    "experiment_config": ExperimentConfig,
    "metrics_report": MetricsReport,
    "agent_decision": AgentDecision,
    "hypothesis_record": HypothesisRecord,
    "student_spec": StudentSpec,
}

# ---------------------------------------------------------------------------
# Discovery helpers (called at collection time for parametrize)
# ---------------------------------------------------------------------------

def _schema_names() -> list[str]:
    """Return schema base names (stem without '.schema') sorted."""
    return sorted(
        p.name.removesuffix(".schema.json")
        for p in SCHEMAS_DIR.glob("*.schema.json")
    )


def _fixture_paths() -> list[Path]:
    """Return all JSON fixture paths sorted."""
    return sorted(FIXTURES_DIR.glob("*.json"))


def _schema_for_fixture(fixture_path: Path) -> str | None:
    """Return the schema base name for a fixture, or None if unmatched.

    Uses longest-prefix matching so 'experiment_config_foo' beats 'experiment'
    if both were schema names (guards against future naming collisions).
    """
    stem = fixture_path.stem
    candidates = [
        s for s in _schema_names()
        if stem == s or stem.startswith(s + "_")
    ]
    return max(candidates, key=len) if candidates else None


def _load_schema(schema_name: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / f"{schema_name}.schema.json"
    return json.loads(path.read_text())


def _jsonschema_validate(instance: Any, schema: dict[str, Any], label: str) -> None:
    """Validate instance against schema using the validator class declared in $schema."""
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        messages = "\n".join(f"  [{list(e.path)}] {e.message}" for e in errors)
        pytest.fail(f"JSON Schema validation failed for {label}:\n{messages}")


# ---------------------------------------------------------------------------
# Coverage tests — run once, not parametrised
# ---------------------------------------------------------------------------

def test_all_fixtures_have_matching_schema() -> None:
    """Every fixture must resolve to a schema. Orphan fixtures are an error."""
    unmatched = [
        p.name for p in _fixture_paths() if _schema_for_fixture(p) is None
    ]
    assert not unmatched, (
        "Fixtures with no matching schema "
        "(fixture filename must start with '<schema_name>_'):\n  "
        + "\n  ".join(unmatched)
    )


def test_all_schemas_have_at_least_one_fixture() -> None:
    """Every schema must have at least one fixture. Uncovered schemas are an error."""
    coverage: dict[str, list[str]] = {s: [] for s in _schema_names()}
    for p in _fixture_paths():
        s = _schema_for_fixture(p)
        if s in coverage:
            coverage[s].append(p.name)
    uncovered = [s for s, fs in coverage.items() if not fs]
    assert not uncovered, (
        "Schemas with no fixture in tests/fixtures/:\n  "
        + "\n  ".join(uncovered)
    )


def test_schema_model_map_covers_all_schemas() -> None:
    """SCHEMA_MODEL_MAP must have an entry for every discovered schema.

    If this test fails, add the new schema's Pydantic model to
    SCHEMA_MODEL_MAP at the top of this file.
    """
    missing = [s for s in _schema_names() if s not in SCHEMA_MODEL_MAP]
    assert not missing, (
        "Schemas not covered by SCHEMA_MODEL_MAP in test_schemas.py "
        "(add the Pydantic model for each):\n  "
        + "\n  ".join(missing)
    )


# ---------------------------------------------------------------------------
# Per-fixture parametrised tests
# ---------------------------------------------------------------------------

_FIXTURE_PATHS = _fixture_paths()
_FIXTURE_IDS = [p.stem for p in _FIXTURE_PATHS]


@pytest.mark.parametrize("fixture_path", _FIXTURE_PATHS, ids=_FIXTURE_IDS)
def test_jsonschema_validation(fixture_path: Path) -> None:
    """Fixture validates against its matching JSON Schema (Draft 2020-12)."""
    schema_name = _schema_for_fixture(fixture_path)
    if schema_name is None:
        pytest.skip("No schema match — covered by test_all_fixtures_have_matching_schema")

    schema = _load_schema(schema_name)
    instance = json.loads(fixture_path.read_text())
    _jsonschema_validate(instance, schema, label=fixture_path.name)


@pytest.mark.parametrize("fixture_path", _FIXTURE_PATHS, ids=_FIXTURE_IDS)
def test_pydantic_roundtrip(fixture_path: Path) -> None:
    """Fixture loads through Pydantic, serialises back to JSON, and re-validates.

    Checks three things:
      1. Pydantic accepts the fixture (type coercion, cross-field validators).
      2. The serialised form is still valid JSON Schema (no Pydantic-only fields
         leaked into the output, no format changes that break the schema).
      3. Re-loading the serialised form produces an identical model state
         (round-trip is lossless).
    """
    schema_name = _schema_for_fixture(fixture_path)
    if schema_name is None:
        pytest.skip("No schema match — covered by test_all_fixtures_have_matching_schema")

    model_cls = SCHEMA_MODEL_MAP.get(schema_name)
    if model_cls is None:
        pytest.skip(
            f"No Pydantic model mapped for '{schema_name}' — "
            "covered by test_schema_model_map_covers_all_schemas"
        )

    original = json.loads(fixture_path.read_text())

    # Step 1: Pydantic accepts the fixture.
    instance = model_cls.model_validate(original)

    # Step 2: Serialised form re-validates against JSON Schema.
    serialised = json.loads(instance.model_dump_json())
    schema = _load_schema(schema_name)
    _jsonschema_validate(serialised, schema, label=f"{fixture_path.name} (round-tripped)")

    # Step 3: Re-loading the serialised form produces an identical model state.
    instance2 = model_cls.model_validate(serialised)
    assert instance.model_dump() == instance2.model_dump(), (
        f"Round-trip produced a different model state for {fixture_path.name}.\n"
        f"This usually means a field serialises to a form Pydantic coerces differently "
        f"on reload (e.g. an enum value vs its string repr)."
    )
