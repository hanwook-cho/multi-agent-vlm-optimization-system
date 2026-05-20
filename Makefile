.DEFAULT_GOAL := help

.PHONY: help verify-phase-0-auto

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'

verify-phase-0-auto: ## Validate all schemas and fixtures (Task 1.3 acceptance check)
	python3 -m pytest tests/ -v
	python3 -c "import sys; sys.path.insert(0, '.'); \
		from schemas import DeviceDescriptor, ExperimentConfig, MetricsReport, AgentDecision, HypothesisRecord; \
		print('All schema imports: OK')"
