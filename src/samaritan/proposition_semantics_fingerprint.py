"""Canonical fingerprints for immutable proposition semantics definitions."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from samaritan.proposition_semantics import (
    PropositionSemantics,
    PropositionRoleDefinition,
    SemanticParameterDefinition,
    TemporalConstraintDefinition,
    TemporalScopeDefinition,
)


SEMANTICS_FINGERPRINT_ALGORITHM = "sha256-v1"


def _canonical_text(value: str | None) -> str | None:
    """Normalize definition text to Unicode NFC without changing its meaning."""
    if value is None:
        return None
    return unicodedata.normalize("NFC", value)


def _role_payload(role: PropositionRoleDefinition) -> dict[str, Any]:
    return {
        "role_name": _canonical_text(role.role_name),
        "role_kind": _canonical_text(role.role_kind),
        "cardinality": _canonical_text(role.cardinality),
        "meaning": _canonical_text(role.meaning),
        "entity_type": _canonical_text(role.entity_type),
        "event_type": _canonical_text(role.event_type),
        "required": role.required,
        "schema_version": role.schema_version,
    }


def _temporal_constraint_payload(
    constraint: TemporalConstraintDefinition,
) -> dict[str, Any]:
    return {
        "earlier_role": _canonical_text(constraint.earlier_role),
        "later_role": _canonical_text(constraint.later_role),
        "relation": _canonical_text(constraint.relation),
        "max_interval_seconds": constraint.max_interval_seconds,
        "schema_version": constraint.schema_version,
    }


def _temporal_scope_payload(
    scope: TemporalScopeDefinition | None,
) -> dict[str, Any] | None:
    if scope is None:
        return None
    return {
        "start_role": _canonical_text(scope.start_role),
        "end_role": _canonical_text(scope.end_role),
        "start_inclusive": scope.start_inclusive,
        "end_inclusive": scope.end_inclusive,
        "schema_version": scope.schema_version,
    }


def _parameter_payload(
    parameter: SemanticParameterDefinition,
) -> dict[str, Any]:
    default_value = parameter.default_value
    if isinstance(default_value, str):
        default_value = _canonical_text(default_value)
    return {
        "parameter_name": _canonical_text(parameter.parameter_name),
        "value_type": _canonical_text(parameter.value_type),
        "required": parameter.required,
        "default_value": default_value,
        "meaning": _canonical_text(parameter.meaning),
        "schema_version": parameter.schema_version,
    }


def canonical_proposition_semantics_payload(
    semantics: PropositionSemantics,
) -> dict[str, Any]:
    """Return the complete, order-independent semantic payload for fingerprinting.

    Human-readable description text is intentionally excluded. Construction
    bindings are not part of PropositionSemantics and must remain independent
    from the meaning represented by this fingerprint.
    """
    if type(semantics) is not PropositionSemantics:
        raise TypeError("semantics must be PropositionSemantics")

    roles = sorted(
        (_role_payload(role) for role in semantics.roles),
        key=lambda item: item["role_name"],
    )
    constraints = sorted(
        (
            _temporal_constraint_payload(item)
            for item in semantics.temporal_constraints
        ),
        key=lambda item: (
            item["earlier_role"],
            item["later_role"],
            item["relation"],
            -1 if item["max_interval_seconds"] is None else item["max_interval_seconds"],
            item["schema_version"],
        ),
    )
    parameters = sorted(
        (_parameter_payload(item) for item in semantics.parameters),
        key=lambda item: item["parameter_name"],
    )

    role_names = [item["role_name"] for item in roles]
    parameter_names = [item["parameter_name"] for item in parameters]
    constraint_keys = [
        (item["earlier_role"], item["later_role"])
        for item in constraints
    ]
    if len(role_names) != len(set(role_names)):
        raise ValueError("role names collide after Unicode NFC normalization")
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("parameter names collide after Unicode NFC normalization")
    if set(role_names) & set(parameter_names):
        raise ValueError("role and parameter names collide after Unicode NFC normalization")
    if len(constraint_keys) != len(set(constraint_keys)):
        raise ValueError("temporal constraints collide after Unicode NFC normalization")

    return {
        "fingerprint_algorithm": SEMANTICS_FINGERPRINT_ALGORITHM,
        "proposition_type": _canonical_text(semantics.proposition_type),
        "semantics_version": semantics.semantics_version,
        "schema_version": semantics.schema_version,
        "roles": roles,
        "temporal_constraints": constraints,
        "temporal_scope": _temporal_scope_payload(semantics.temporal_scope),
        "parameters": parameters,
    }


def canonical_proposition_semantics_json(
    semantics: PropositionSemantics,
) -> str:
    """Serialize a semantics definition into deterministic canonical JSON."""
    payload = canonical_proposition_semantics_payload(semantics)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def proposition_semantics_fingerprint(
    semantics: PropositionSemantics,
) -> str:
    """Return a versioned SHA-256 fingerprint of a semantics definition."""
    canonical_json = canonical_proposition_semantics_json(semantics)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"{SEMANTICS_FINGERPRINT_ALGORITHM}:{digest}"
