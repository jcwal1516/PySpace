# PySpace 0.1.0 release-candidate verification

Verified: 2026-08-04T02:00:44Z

Host: macOS arm64

Primary Python: 3.11.14

R: 4.5.2

Pinned SPACE commit: `94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8`

## Executed gates

| Command | Result |
| --- | --- |
| `PATH="$PWD/.venv/bin:$PATH" make check` | PASS: Ruff format/lint clean; mypy clean across 86 checked files; 201 tests passed; 84.10% overall and 92.08% parity/public-I/O kernel coverage. |
| `SPACE_R_REPO=/tmp/pyspace-r-upstream.nwDhgo PATH="$PWD/.venv/bin:$PATH" make parity` | PASS: 59 tests against the pristine pinned R checkout, including linked object/scalar transMI input. |
| `.venv/bin/python -m pyspace.parity --r-repo /tmp/pyspace-r-upstream.nwDhgo --json --output PARITY_REPORT.json` | PASS: 40/40 declarative checks; score 1.0. |
| `PATH="$PWD/.venv/bin:$PATH" make test-all` | PASS: 201 tests. |
| `PATH="$PWD/.venv/bin:$PATH" make build` | PASS: wheel and sdist built; Twine and member inspection passed. |
| `.venv/bin/python scripts/smoke_test_wheel.py dist/pyspace_analysis-0.1.0-py3-none-any.whl` | PASS: clean install and CLI/import smoke on Python 3.11.14. |
| `/usr/local/bin/python3.12 scripts/smoke_test_wheel.py dist/pyspace_analysis-0.1.0-py3-none-any.whl` | PASS: Python 3.12.9. |
| `/opt/homebrew/bin/python3.13 scripts/smoke_test_wheel.py dist/pyspace_analysis-0.1.0-py3-none-any.whl` | PASS: Python 3.13.11. |
| `/opt/homebrew/bin/python3.14 scripts/smoke_test_wheel.py dist/pyspace_analysis-0.1.0-py3-none-any.whl` | PASS: Python 3.14.4. |
| `PATH="$PWD/.venv/bin:$PATH" make docs` | PASS: strict MkDocs build; Material emitted its informational MkDocs 2.0 advisory. |
| `PYSPACE_REQUIRE_COMMUNITY=1 .venv/bin/pytest -q tests/test_networks.py` | PASS: 5 tests with igraph, Leiden, and Infomap installed. |
| `PATH="$PWD/.venv/bin:$PATH" make benchmark` | PASS: output checksums matched before five timing samples per workload; see `BENCHMARK_REPORT.json`. |
| `.venv/bin/python scripts/check_public_tree.py` | PASS: no prohibited files, caches, study-specific paths, or unexplained oversized artifacts. |
| `.venv/bin/python scripts/write_data_manifest.py --check` | PASS. |
| `.venv/bin/python scripts/write_dependency_inventory.py --check` | PASS. |
| `.venv/bin/python -m pip_audit` | PASS: no known vulnerabilities; the unpublished local `pyspace-analysis` distribution was skipped. |
| checksum-verified `actionlint` 1.7.12 | PASS: all workflow files valid. |
| checksum-verified `gitleaks dir --no-banner --redact .` 8.30.1 | PASS: no leaks found. |
| `PATH="$PWD/.venv/bin:$PATH" make audit` | PASS: public-tree, data, dependency, vulnerability, license, and final artifact-manifest checks. |

The first local scanner bootstrap attempt stopped before scanning because the
downloaded archive had been given a filename different from its checksum-list
entry. The corrected checksum-verified invocation above passed.

## Not locally verified

- Linux and Windows execution require the configured GitHub Actions matrix.
- CodeQL requires the configured hosted GitHub job; no local `codeql` executable
  was available.

Public repository target: `https://github.com/jcwal1516/PySpace`. Hosted workflow
results are not represented in this local verification record. No PyPI upload,
tag, or GitHub release was performed.
