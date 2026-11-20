from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from samaritan.proposition_contracts import (
    Proposition,
    PropositionFieldProvenance,
    PropositionRoleValue,
    PropositionTemporalScope,
    deterministic_proposition_id,
)
from samaritan.proposition_semantics import build_default_proposition_semantics_registry


def make_proposition():
    start = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    times = (start, start + timedelta(seconds=90), start + timedelta(seconds=180))
    account = str(uuid4())
    events = tuple(str(uuid4()) for _ in range(3))
    roles = (
        PropositionRoleValue("account", "entity", account),
        PropositionRoleValue("failure_event", "event", events[0], times[0]),
        PropositionRoleValue("success_event", "event", events[1], times[1]),
        PropositionRoleValue("host_activity_event", "event", events[2], times[2]),
    )
    scope = PropositionTemporalScope(times[0], times[2], True, True)
    provenance = (
        PropositionFieldProvenance("account", events, tuple(str(uuid4()) for _ in range(3))),
        PropositionFieldProvenance("failure_event", (events[0],), (str(uuid4()),)),
        PropositionFieldProvenance("success_event", (events[1],), (str(uuid4()),)),
        PropositionFieldProvenance("host_activity_event", (events[2],), (str(uuid4()),)),
    )
    proposition_id = deterministic_proposition_id(
        "account_authentication_sequence", 1, roles, scope
    )
    return Proposition(
        proposition_id=proposition_id,
        proposition_type="account_authentication_sequence",
        semantics_version=1,
        roles=roles,
        temporal_scope=scope,
        field_provenance=provenance,
        construction_id="account_authentication_sequence",
        construction_version=1,
    )


def test_structured_proposition_validates_against_exact_semantics():
    proposition = make_proposition()
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    proposition.validate_against(semantics)


def test_proposition_identity_is_deterministic_and_ignores_construction_history():
    proposition = make_proposition()
    same_identity = Proposition(
        proposition_id=proposition.proposition_id,
        proposition_type=proposition.proposition_type,
        semantics_version=proposition.semantics_version,
        roles=proposition.roles,
        temporal_scope=proposition.temporal_scope,
        field_provenance=proposition.field_provenance,
        construction_id="different-construction-history",
        construction_version=9,
    )
    assert same_identity.proposition_id == proposition.proposition_id


def test_temporal_scope_must_match_semantic_boundaries():
    proposition = make_proposition()
    wrong_scope = PropositionTemporalScope(
        proposition.temporal_scope.starts_at + timedelta(seconds=1),
        proposition.temporal_scope.ends_at,
    )
    invalid = Proposition(
        proposition_id=str(uuid4()),
        proposition_type=proposition.proposition_type,
        semantics_version=1,
        roles=proposition.roles,
        temporal_scope=wrong_scope,
        field_provenance=proposition.field_provenance,
        construction_id=proposition.construction_id,
        construction_version=1,
    )
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    with pytest.raises(ValueError, match="temporal scope"):
        invalid.validate_against(semantics)


def test_required_temporal_interval_is_enforced():
    proposition = make_proposition()
    roles = tuple(
        PropositionRoleValue(
            role.role_name,
            role.role_kind,
            role.value_id,
            (
                role.occurred_at + timedelta(seconds=400)
                if role.role_name == "success_event"
                else role.occurred_at
            ),
        )
        for role in proposition.roles
    )
    invalid = Proposition(
        proposition_id=str(uuid4()),
        proposition_type=proposition.proposition_type,
        semantics_version=1,
        roles=roles,
        temporal_scope=proposition.temporal_scope,
        field_provenance=proposition.field_provenance,
        construction_id=proposition.construction_id,
        construction_version=1,
    )
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    with pytest.raises(ValueError, match="interval"):
        invalid.validate_against(semantics)


def test_every_role_requires_field_level_provenance():
    proposition = make_proposition()
    with pytest.raises(ValueError, match="field-level provenance"):
        Proposition(
            proposition_id=proposition.proposition_id,
            proposition_type=proposition.proposition_type,
            semantics_version=1,
            roles=proposition.roles,
            temporal_scope=proposition.temporal_scope,
            field_provenance=proposition.field_provenance[:-1],
            construction_id=proposition.construction_id,
            construction_version=1,
        )

def test_proposition_identity_must_match_semantic_content():
    proposition = make_proposition()
    invalid = Proposition(
        proposition_id=str(uuid4()),
        proposition_type=proposition.proposition_type,
        semantics_version=proposition.semantics_version,
        roles=proposition.roles,
        temporal_scope=proposition.temporal_scope,
        field_provenance=proposition.field_provenance,
        construction_id=proposition.construction_id,
        construction_version=proposition.construction_version,
    )
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    with pytest.raises(ValueError, match="deterministic semantic identity"):
        invalid.validate_against(semantics)


def test_event_role_provenance_must_name_the_role_event():
    proposition = make_proposition()
    invalid_provenance = tuple(
        PropositionFieldProvenance(
            item.field_name,
            (str(uuid4()),) if item.field_name == "failure_event" else item.event_ids,
            item.observation_ids,
            item.relationship_ids,
        )
        for item in proposition.field_provenance
    )
    with pytest.raises(ValueError, match="exact event"):
        Proposition(
            proposition_id=proposition.proposition_id,
            proposition_type=proposition.proposition_type,
            semantics_version=proposition.semantics_version,
            roles=proposition.roles,
            temporal_scope=proposition.temporal_scope,
            field_provenance=invalid_provenance,
            construction_id=proposition.construction_id,
            construction_version=proposition.construction_version,
        )

