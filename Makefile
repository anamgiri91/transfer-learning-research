.PHONY: help setup data splits bench indomain analysis report test lint check-private clean verify verify-online verify-repro verify-repro-full
.DEFAULT_GOAL := help

PY ?= python
# eva71_3c was the pre-Amendment-1 target and has no dataset. The study runs
# on eva71_2a; a default that names a file nobody can build is a broken target
# dressed up as a configurable one.
TARGET ?= eva71_2a

# No default: look the target up in ChEMBL and pass it explicitly, e.g.
#   make data TARGET=eva71_3c CHEMBL_ID=CHEMBLxxxxx
CHEMBL_ID ?=

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"
	nbstripout --install || true

data:  ## Build the evaluation set from the OpenBind release
	@# Amendment 1 replaced the ChEMBL fetch with the OpenBind Zenodo release.
	@# fetch_data.py/curate_data.py are the pre-Amendment-1 path and are kept
	@# for the in-domain corpus, which IS from ChEMBL (see `make indomain`).
	$(PY) scripts/prepare_openbind.py

splits:  ## Build split files and run leakage assertions
	$(PY) scripts/build_splits.py --target $(TARGET)

bench:  ## Run every arm reported in the paper, across seeds, sizes and splits
	@# This used to loop over config/*.yaml through run_benchmark.py. That
	@# harness describes a study that was never executed -- LightGBM, a D-MPNN,
	@# 500-compound budgets, and the pre-Amendment-2 T1/T2 assignment -- and it
	@# failed on its first config. It is retired; see plan.md Amendment 3.
	$(PY) scripts/run_arms.py --arms B0 B1 B2 T1 T0r --splits scaffold random butina
	$(PY) scripts/run_arms.py --arms T2 --splits scaffold
	@# T4/T5 need the encoders from `make indomain` and are skipped if absent.
	@if [ -f models/indomain_T4.pt ]; then \
	  $(PY) scripts/run_arms.py --arms T4 T5 --splits scaffold random butina; \
	  $(PY) scripts/run_arms.py --arms T4c T4r T5c T5r --splits scaffold; \
	else echo "skipping T4/T5: run \`make indomain\` first"; fi

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

analysis:  ## Re-derive every analysis table from results/metrics/
	$(PY) scripts/analyse_cliffs.py
	$(PY) scripts/analyse_indomain.py
	$(PY) scripts/analyse_tuning.py
	$(PY) scripts/analyse_endpoints.py
	$(PY) scripts/analyse_h2.py

report:  ## Regenerate figures and tables from results/metrics/
	$(PY) scripts/make_report.py

test:  ## Run the test suite
	$(PY) -m pytest

verify:  ## Run tests, check tables are fresh, verify every claim and citation
	$(PY) -m pytest -q
	$(PY) scripts/render_manuscript_tables.py --check
	$(PY) scripts/audit_splits.py
	$(PY) scripts/verify_manuscript.py
	$(PY) scripts/verify_citations.py
	$(PY) scripts/verify_consistency.py

verify-repro:  ## Reconstruct the offline stages in an isolated worktree and diff
	@# Not part of `verify`: it takes minutes, where `verify` runs constantly.
	@# Runs in a throwaway git worktree at an immutable ref, so it never writes
	@# to your tree and a failed run cannot become the next run baseline.
	$(PY) scripts/verify_reproducibility.py --tier fast

verify-repro-full:  ## As above, plus the transfer arms (~25 min)
	$(PY) scripts/verify_reproducibility.py --tier full

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
