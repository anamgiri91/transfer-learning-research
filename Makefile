.PHONY: help setup data splits bench indomain report test lint check-private clean verify verify-online
.DEFAULT_GOAL := help

PY ?= python
TARGET ?= eva71_3c

# No default: look the target up in ChEMBL and pass it explicitly, e.g.
#   make data TARGET=eva71_3c CHEMBL_ID=CHEMBLxxxxx
CHEMBL_ID ?=

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"
	nbstripout --install || true

data:  ## Fetch + curate raw sources. Requires CHEMBL_ID=<id>
	@test -n "$(CHEMBL_ID)" || { echo "Set CHEMBL_ID=<ChEMBL target id>. See data/raw/README.md"; exit 1; }
	$(PY) scripts/fetch_data.py --target-chembl-id $(CHEMBL_ID) --name $(TARGET)_chembl
	$(PY) scripts/curate_data.py --target $(TARGET)

splits:  ## Build split files and run leakage assertions
	$(PY) scripts/build_splits.py --target $(TARGET)

bench:  ## Run every arm in config/ across seeds and training sizes
	@for cfg in config/arm_*.yaml; do \
	  echo "== $$cfg"; $(PY) scripts/run_benchmark.py --config $$cfg || exit 1; \
	done

indomain:  ## Fetch, curate and pretrain the in-domain 3C/3CL corpus (arms T4/T5)
	$(PY) scripts/fetch_indomain.py
	$(PY) scripts/prepare_indomain.py
	@for a in T4 T5; do \
	  $(PY) scripts/pretrain_indomain.py --arm $$a || exit 1; \
	  $(PY) scripts/pretrain_indomain.py --arm $$a --decontaminate || exit 1; \
	done
	@# Post-condition. A run of this loop once produced nothing at all and still
	@# reported success, because the failing argument was swallowed by a log
	@# filter. Exit status is not evidence that the artefacts exist; check them.
	@missing=""; for m in indomain_T4 indomain_T5 indomain_T4_clean indomain_T5_clean; do \
	  test -f models/$$m.pt -a -f models/$$m.json || missing="$$missing $$m"; \
	done; \
	if [ -n "$$missing" ]; then \
	  echo "FAILED: expected encoders were not written:$$missing"; exit 1; fi; \
	echo "all four in-domain encoders present."

report:  ## Regenerate figures and tables from results/metrics/
	$(PY) scripts/make_report.py

test:  ## Run the test suite
	$(PY) -m pytest

verify:  ## Run tests, check tables are fresh, verify every claim and citation
	$(PY) -m pytest -q
	$(PY) scripts/render_manuscript_tables.py --check
	$(PY) scripts/verify_manuscript.py
	$(PY) scripts/verify_citations.py

verify-online:  ## verify, plus check that every cited URL still resolves
	$(MAKE) verify
	$(PY) scripts/verify_citations.py --online

lint:  ## Lint and format-check
	ruff check src tests scripts
	ruff format --check src tests scripts

check-private:  ## Verify nothing private is staged for commit. Run before every push.
	@echo "Checking the git index for private content..."
	@fail=0; \
	staged=$$(git ls-files --cached); \
	if echo "$$staged" | grep -E '^private/' | grep -qv '^private/README\.md$$'; then \
	  echo "REFUSING: private/ content is staged:"; \
	  echo "$$staged" | grep -E '^private/' | grep -v '^private/README\.md$$'; fail=1; fi; \
	if echo "$$staged" | grep -vE '(^|/)\.env\.example$$' | grep -qE '(^|/)\.env($$|\.)|\.(pem|key)$$|_credentials\.json$$'; then \
	  echo "REFUSING: credential-shaped file is staged:"; \
	  echo "$$staged" | grep -vE '(^|/)\.env\.example$$' | grep -E '(^|/)\.env($$|\.)|\.(pem|key)$$|_credentials\.json$$'; fail=1; fi; \
	if echo "$$staged" | grep -qE '^data/raw/.*\.(csv|parquet|sdf|tsv|zip)$$'; then \
	  echo "REFUSING: data payload is staged:"; \
	  echo "$$staged" | grep -E '^data/raw/.*\.(csv|parquet|sdf|tsv|zip)$$'; fail=1; fi; \
	if [ $$fail -eq 1 ]; then exit 1; fi; \
	echo "clean."

clean:  ## Remove regenerable artefacts (never touches data/raw or private/)
	rm -rf data/interim/* results/figures/* results/tables/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
