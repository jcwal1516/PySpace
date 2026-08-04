PYTHON ?= python
PYTHON_PATHS = src tests scripts benchmarks examples

.PHONY: install-dev check parity test-all build docs audit benchmark

install-dev:
	$(PYTHON) -m pip install -e '.[dev,docs,community]'

check:
	$(PYTHON) -m ruff format --check $(PYTHON_PATHS)
	$(PYTHON) -m ruff check $(PYTHON_PATHS)
	$(PYTHON) -m mypy $(PYTHON_PATHS)
	$(PYTHON) -m pytest -q --cov=pyspace --cov-report=term-missing --cov-report=json:coverage.json
	$(PYTHON) scripts/check_kernel_coverage.py coverage.json

parity:
	$(PYTHON) -m pytest -q tests/test_core_parity.py tests/test_diversity_parity.py tests/test_live_r_parity.py tests/test_loaders_parity.py tests/test_merge_objects_parity.py tests/test_parameters.py tests/test_parity_harness.py tests/test_patterns_parity.py tests/test_tutorial_parity.py

test-all:
	$(PYTHON) -m pytest -q

build:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
	$(PYTHON) scripts/inspect_distribution.py dist/*
	$(PYTHON) scripts/smoke_test_wheel.py $$(find dist -maxdepth 1 -name '*.whl' -print -quit)

docs:
	$(PYTHON) -m mkdocs build --strict

audit:
	$(PYTHON) scripts/check_public_tree.py
	$(PYTHON) scripts/write_data_manifest.py --check
	$(PYTHON) scripts/write_dependency_inventory.py --check
	$(PYTHON) -m pip_audit
	$(PYTHON) -m piplicenses --format=markdown --with-urls
	$(PYTHON) scripts/write_artifact_manifest.py --check

benchmark:
	$(PYTHON) benchmarks/compare_r_python.py
