SNP Dashboard — Test Execution Guide

This repo includes unit and integration tests configured via pytest. Follow the steps below to set up the environment and run tests.

## Prerequisites

- Python 3.11+
- Optional: a virtual environment (recommended)

## Setup

- Create and activate a virtual environment
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows (PowerShell): `py -3.11 -m venv .venv; .venv\\Scripts\\Activate.ps1`
- Install dependencies
  - Easiest: `pip install -r requirements.txt`
  - Alternative (editable + dev extras): `pip install -e .[dev]`

Note: A `.env` file with `FRED_API_KEY=...` exists, but current tests do not call external APIs and do not require it.

## Running Tests

- All tests with coverage: `pytest`
  - Configured in `pyproject.toml` to collect coverage for `src` and `app` and emit reports.
- Unit tests only: `pytest tests/unit`
- Integration tests only: `pytest tests/integration`
- Quiet output: `pytest -q`
- Verbose output: `pytest -vv`

Tip: If you installed only from `requirements.txt` (not `-e .`), add `PYTHONPATH=$PWD` to make `src` importable, e.g. `PYTHONPATH=$PWD pytest`.

## Coverage Artifacts

Running pytest (with the default config) produces:

- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`
- Terminal summary: missing lines per file (skip-covered)

## Useful Paths

- Tests root: `tests`
- Unit tests: `tests/unit`
- Integration tests: `tests/integration`
- Pytest config: `pyproject.toml`

## Troubleshooting

- If pytest is not found: ensure the venv is activated (`source .venv/bin/activate`).
- If import errors occur: confirm install completed without errors and that you’re running from the repo root.
- If coverage HTML isn’t generated: ensure you’re not overriding addopts; run `pytest --cov=src --cov=app --cov-report=html --cov-report=term-missing:skip-covered`.
- If you hit plugin import errors (e.g., unrelated system pytest plugins):
  - Minimal run (no coverage): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q -o addopts=`
  - With coverage (requires pytest-cov installed): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -p pytest_cov --cov=src --cov=app --cov-report=term-missing:skip-covered --cov-report=html --cov-report=xml`

## Example Commands

- Fresh setup + run all tests:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `PYTHONPATH=$PWD pytest`

- Only unit tests (quiet):
  - `PYTHONPATH=$PWD pytest -q tests/unit`

- Only integration tests (verbose):
  - `PYTHONPATH=$PWD pytest -vv tests/integration`
