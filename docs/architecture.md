# Architecture Notes

## Architecture

Samaritan begins as a modular monolith running locally.

Initial technology:
- Python for application logic.
- SQLite for local persistence.
- pytest for automated verification.

## Logical capabilities

1. Perception — acquire, validate, and normalize authorized information.
2. Memory — preserve observations, entities, events, and relationships.
3. Reasoning — construct structured propositions, evaluate evidence, and compare explanations.
4. Prediction — estimate probabilities only after deterministic evidence evaluation is trustworthy.
5. Decision — prioritize risks and recommend responses under explicit governance.
6. Controlled action — execute only permissioned and auditable operations.
7. Adaptation — evaluate outcomes and improve analytical quality.

These are logical boundaries, not separate services. Implement them incrementally as the project earns the complexity.

## Evidence and reasoning pipeline

The current analytical boundary is:

JSON/JSONL -> validation -> normalization -> SQLite -> retrieval -> correlation -> hypothesis generation -> EvidenceSet -> deterministic proposition construction -> deterministic criterion evaluation -> stance-neutral findings -> alternative-aware diagnosticity synthesis

M2.4 keeps proposition semantics separate from construction definitions. Semantics define what proposition fields mean and what makes a proposition valid. Construction definitions bind those fields to registered source structures. Every constructed field retains provenance.

Criterion findings are stance-neutral and reusable. Versioned bearing definitions interpret findings against a focal proposition and each declared alternative. Diagnosticity synthesis is ordinal and deterministic; it does not calculate probabilities.

M7 adds append-only EvidenceSet selection records to SQLite. Proposition construction persists the exact selection before emitting a proposition, and provenance validation checks the manifest against that stored record. The database migrates from schema version 1 to 2 without rewriting existing source records. Proposition and evaluation records themselves are not yet persisted. See m2.4-analytical-foundation.md for the complete model and limitations.

## Evidence integrity

The pipeline must preserve original source information, validate references, reject ambiguous construction, and never silently discard legitimate repeated events. Shared identifiers do not establish common actors, ownership, authorization, or malicious intent.
