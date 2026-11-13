# Architecture Notes

## Initial architecture

Samaritan begins as a modular monolith running locally.

Initial technology:
- Python for application logic.
- SQLite for local persistence.
- pytest for automated verification.

## Logical capabilities

1. Perception — acquire, validate, and normalize authorized information.
2. Memory — preserve observations, entities, events, and relationships.
3. Reasoning — correlate evidence and evaluate hypotheses.
4. Prediction — produce measurable forecasts with uncertainty.
5. Decision — prioritize risks and recommend responses.
6. Controlled action — execute only permissioned and auditable operations.
7. Adaptation — evaluate outcomes and improve analytical quality.

These are logical boundaries, not separate services. We will implement
them incrementally as the project earns the complexity.

## First implementation milestone

Build a synthetic cybersecurity telemetry pipeline:

JSON/JSONL -> validation -> normalization -> SQLite -> retrieval

The pipeline must preserve original source information and must not
silently discard legitimate repeated events.
