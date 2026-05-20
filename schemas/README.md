# Schemas

JSON Schema (Draft 2020-12) contracts for all agent/service interfaces,
plus Pydantic v2 models for Python-side validation.

**Schema rationale and field semantics:** see `docs/HLD.md` §6.

**Example instances:** see `tests/fixtures/`.

## Files

| Schema | Pydantic module | Description |
|---|---|---|
| `device_descriptor.schema.json` | `devices.py` | Hardware device capabilities and quirks |
| `experiment_config.schema.json` | `experiments.py` | A single experiment to run |
| `metrics_report.schema.json` | `experiments.py` | Measured results from a completed experiment |
| `agent_decision.schema.json` | `agents.py` | Logged decision record for any agent action |
| `hypothesis_record.schema.json` | `agents.py` | Research Analyst Agent output — implementation kit for a technique |

Agent I/O contracts (`search_strategist_input`, `search_strategist_output`,
`research_analyst_input`) are added in Phase 1 when the agents are built.
