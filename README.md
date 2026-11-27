# Samaritan

Samaritan is a local-first, evidence-driven intelligence platform.

Its initial development domain is cybersecurity. The system is designed to ingest authorized telemetry, preserve its provenance, correlate observations, construct structured propositions from validated evidence, and produce explainable assessments.

## Development principles

- Local-first operation.
- Evidence before conclusions.
- Preserve source data and field-level provenance.
- Distinguish observations, inferences, propositions, and hypotheses.
- Deterministic rules for explicit, verifiable conditions.
- Least privilege, explicit authorization, and auditable operations.
- Test every important capability.
- Keep probability estimation and consequential decisions separate from deterministic evidence validation.

## Initial development environment

- Python 3.14+
- SQLite
- pytest

## Project structure

- src/samaritan/ — application package
- tests/ — automated verification
- docs/ — architecture and engineering decisions

## Setup

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install development dependencies:

    python -m pip install pytest

Run tests:

    pytest

## Current milestone

Milestone M2.4 establishes structured proposition semantics, finite deterministic construction definitions, field-level provenance, stance-neutral criterion findings, versioned alternative-aware evaluation rules, and deterministic diagnosticity synthesis.

See docs/m2.4-analytical-foundation.md for the contracts, initial authentication-sequence construction, reasoning model, limitations, and deliberate non-goals.

The initial lab uses synthetic data only. No real systems or third-party environments are targeted.
