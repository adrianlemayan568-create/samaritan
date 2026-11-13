# Samaritan

Samaritan is a local-first, evidence-driven intelligence platform.

Its initial development domain is cybersecurity. The system is designed
to ingest authorized telemetry, preserve its provenance, correlate
observations, and produce explainable assessments.

## Development principles

- Local-first operation.
- Evidence before conclusions.
- Preserve source data and provenance.
- Distinguish observations, inferences, and hypotheses.
- Use least privilege and explicit authorization.
- Audit consequential operations.
- Test every important capability.

## Initial development environment

- Python 3.14+
- SQLite
- pytest

## Project structure

- `src/samaritan/` — application package
- `tests/` — automated tests
- `docs/` — architecture and engineering documentation

## Setup

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install development dependencies:

    python -m pip install pytest

Run tests:

    pytest

## Current milestone

Milestone 001: Intelligence Foundation — Cyber Telemetry Ingestion.

The initial lab uses synthetic data only. No real systems or third-party
environments are targeted.
