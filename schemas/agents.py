from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator, field_validator


class AgentName(str, Enum):
    SEARCH_STRATEGIST = "search_strategist"
    RESEARCH_ANALYST = "research_analyst"


class DecisionType(str, Enum):
    PROPOSE_EXPERIMENTS = "propose_experiments"
    TRIAGE_FAILURE = "triage_failure"
    WRITE_PARETO_RATIONALE = "write_pareto_rationale"
    INGEST_PAPER = "ingest_paper"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Which decision types each agent is allowed to produce.
# Checked by the model_validator below.
_AGENT_DECISION_TYPES: dict[str, set[str]] = {
    AgentName.SEARCH_STRATEGIST: {
        DecisionType.PROPOSE_EXPERIMENTS,
        DecisionType.TRIAGE_FAILURE,
        DecisionType.WRITE_PARETO_RATIONALE,
    },
    AgentName.RESEARCH_ANALYST: {
        DecisionType.INGEST_PAPER,
    },
}


class AgentDecision(BaseModel):
    """Structured log entry for a single LLM-agent decision.

    Every agent action that changes system state or enters the approval queue
    is logged here. The combination of input_hash + output_hash + llm_id
    provides the audit trail for reproducibility and hallucination review.

    agent_name and decision_type must be consistent: search_strategist owns
    propose_experiments / triage_failure / write_pareto_rationale; research_analyst
    owns ingest_paper.
    """

    agent_name: Annotated[AgentName, Field(description="Which agent produced this decision.")]
    decision_type: Annotated[DecisionType, Field(description="What kind of decision this is.")]
    input_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the canonical JSON of the agent's input package.")]
    output_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the canonical JSON of the agent's output artifact.")]
    rationale: Annotated[str, Field(min_length=1, description="Free-form LLM explanation of this decision. Written for a developer reading the decision log.")]
    confidence_flags: Annotated[dict[str, ConfidenceLevel], Field(default_factory=dict, description="Per-field uncertainty flags. Keys are field or concept names; values are low/medium/high. Absent key = no flag raised for that field.")]
    timestamp: Annotated[datetime, Field(description="Timezone-aware timestamp when the agent produced this decision.")]
    llm_id: Annotated[str, Field(min_length=1, description="LLM that produced this decision. Free-form string; canonical format is a HuggingFace repo name (e.g. 'Qwen/Qwen3-Coder-7B') or an API model string (e.g. 'claude-sonnet-4-5'). Not the VLM under evaluation — see ExperimentConfig.model_id for that.")]
    duration_ms: Annotated[int | None, Field(default=None, ge=0, description="Wall-clock time in ms from input submission to output receipt.")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def decision_type_matches_agent(self) -> "AgentDecision":
        allowed = _AGENT_DECISION_TYPES.get(self.agent_name, set())
        if self.decision_type not in allowed:
            agent_label = self.agent_name.replace("_", " ").title() + " Agent"
            # Find which agent actually owns this decision_type, so the error
            # names the responsible party rather than just listing valid values.
            owner = next(
                (a for a, types in _AGENT_DECISION_TYPES.items() if self.decision_type in types),
                None,
            )
            owner_clause = (
                f" — this is a {owner.replace('_', ' ').title()} Agent responsibility"
                if owner else ""
            )
            raise ValueError(
                f"{agent_label} cannot produce decision_type '{self.decision_type}'"
                f"{owner_clause}. Valid types for {agent_label}: {sorted(t.value for t in allowed)}"
            )
        return self


# ---------------------------------------------------------------------------
# HypothesisRecord — Research Analyst Agent output
# ---------------------------------------------------------------------------


class ImplementationDifficulty(str, Enum):
    CONFIG_CHANGE = "config_change"          # Tier 1: auto-runnable by Search Strategist
    MINOR_CODE_CHANGE = "minor_code_change"  # Tier 2: ~few hours human work
    NEW_MODULE = "new_module"                # Tier 2: ~day human work
    MAJOR_REFACTOR = "major_refactor"        # Tier 2: escalate to project-level decision


class ApplicabilityVerdict(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class SourceCitation(BaseModel):
    """Bibliographic reference for a paper or repository.

    At least one of url, arxiv_id, or github_url must be present so the
    citation verifier (HLD §6.4) can confirm the paper exists.
    """

    title: Annotated[str, Field(min_length=1, description="Full paper or repository title as it appears in the source.")]
    authors: Annotated[list[str], Field(min_length=1, description="Author list in order of appearance.")]
    venue: Annotated[str | None, Field(default=None, description="Conference, journal, or workshop (e.g. 'ICLR 2025'). Null for preprints.")]
    year: Annotated[int, Field(ge=2000, description="Publication or preprint year.")]
    arxiv_id: Annotated[str | None, Field(default=None, pattern=r"^\d{4}\.\d{4,7}$", description="arXiv identifier (e.g. '2301.12345'). Enables direct arXiv API citation verification.")]
    github_url: Annotated[str | None, Field(default=None, description="GitHub repository URL, if an official implementation exists.")]
    url: Annotated[str | None, Field(default=None, description="Canonical URL for the paper. The citation verifier fetches this to confirm existence. Required when arxiv_id and github_url are both null.")]

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def at_least_one_link(self) -> "SourceCitation":
        if not any([self.arxiv_id, self.github_url, self.url]):
            raise ValueError(
                "At least one of arxiv_id, github_url, or url must be provided "
                "to enable citation verification (HLD §6.4)."
            )
        return self


class VerbatimExcerpt(BaseModel):
    """A direct quote from the paper grounding a claim in the source text."""

    text: Annotated[str, Field(min_length=1, description="The verbatim quoted passage. Must be an exact quote, not a paraphrase.")]
    location: Annotated[str | None, Field(default=None, description="Where in the paper this appears (e.g. 'Section 3.2, p. 5', 'Table 2'). Helps the human find the passage quickly.")]


class ApplicabilityCheck(BaseModel):
    """Structured verdict on whether this technique applies to the project's setup."""

    requirements: Annotated[list[str], Field(description="Explicit requirements the technique imposes, extracted from the paper.")]
    verdict: Annotated[ApplicabilityVerdict, Field(description="'applicable': requirements met. 'not_applicable': hard requirement unmet. 'uncertain': depends on unknowns.")]
    notes: Annotated[str, Field(min_length=1, description="Reasoning behind the verdict: which requirements are met, which are borderline, what would change the outcome.")]

    model_config = {"use_enum_values": True}


class HypothesisRecord(BaseModel):
    """Research Analyst Agent output — an implementation kit for a technique from a paper.

    Every claim is grounded in verbatim_excerpts so the human can cross-check
    the agent's extraction against the source. Confidence flags surface the
    agent's per-field uncertainty explicitly rather than hiding it.
    """

    title: Annotated[str, Field(min_length=1, description="Short technique name, suitable as a heading.")]
    source_citation: Annotated[SourceCitation, Field(description="Bibliographic reference for the source paper or repository.")]
    claimed_effect: Annotated[str, Field(min_length=1, description="What the paper says the technique does, in 2-3 sentences. In the paper's own terms, grounded in verbatim_excerpts.")]
    verbatim_excerpts: Annotated[list[VerbatimExcerpt], Field(min_length=1, description="Direct quotes from the paper. At least one required. The human uses these to cross-check the extraction without re-reading the full paper.")]
    original_hyperparameters: Annotated[str | None, Field(default=None, description="The paper's reported configuration as free-form text. Null when the paper does not report specific hyperparameters.")]
    reported_results: Annotated[str, Field(min_length=1, description="The paper's claimed numbers and the evaluation setup that produced them. Free-form text; not independently verified.")]
    applicability_check: Annotated[ApplicabilityCheck, Field(description="Structured assessment of whether this technique applies to our model, data, hardware, and quantization setup.")]
    known_failure_modes: Annotated[list[str], Field(description="Limitations the paper discloses, typically from discussion or limitations sections. Empty list if the paper discloses none.")]
    implementation_difficulty: Annotated[ImplementationDifficulty, Field(description="Estimated implementation effort. Determines Tier 1 (auto-runnable) vs Tier 2 (human-implemented) routing. See HLD §6.3.")]
    proposed_codebase_insertion_point: Annotated[str | None, Field(default=None, description="Which file or class this technique would extend (e.g. 'runners/mlx_runner.py, VisionEncoder.forward'). A starting point, not final code. Null when the agent cannot determine a specific location.")]
    confidence_flags: Annotated[dict[str, ConfidenceLevel], Field(default_factory=dict, description="Per-field uncertainty flags. Keys are field names or concept names; values are low/medium/high. Empty means no uncertainty flagged.")]

    model_config = {"use_enum_values": True}
