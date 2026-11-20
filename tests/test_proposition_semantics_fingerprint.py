from dataclasses import replace
import re

import pytest

from samaritan.proposition_semantics import (
    PropositionRoleDefinition,
    PropositionSemantics,
    SemanticParameterDefinition,
    TemporalConstraintDefinition,
    TemporalScopeDefinition,
    build_default_proposition_semantics_registry,
)
from samaritan.proposition_semantics_fingerprint import (
    SEMANTICS_FINGERPRINT_ALGORITHM,
    canonical_proposition_semantics_json,
    canonical_proposition_semantics_payload,
    proposition_semantics_fingerprint,
)


def make_semantics():
    return PropositionSemantics(
        proposition_type="test_sequence",
        semantics_version=3,
        roles=(
            PropositionRoleDefinition(
                role_name="account",
                role_kind="entity",
                cardinality="one",
                entity_type="user_account",
                meaning="The account associated with both events.",
            ),
            PropositionRoleDefinition(
                role_name="start_event",
                role_kind="event",
                cardinality="one",
                event_type="authentication_failure",
                meaning="The initiating authentication failure.",
            ),
            PropositionRoleDefinition(
                role_name="end_event",
                role_kind="event",
                cardinality="one",
                event_type="authentication_success",
                required=True,
                meaning="The subsequent successful authentication.",
            ),
        ),
        temporal_constraints=(
            TemporalConstraintDefinition(
                earlier_role="start_event",
                later_role="end_event",
                max_interval_seconds=120,
            ),
        ),
        temporal_scope=TemporalScopeDefinition(
            start_role="start_event",
            end_role="end_event",
            start_inclusive=True,
            end_inclusive=False,
        ),
        parameters=(
            SemanticParameterDefinition(
                parameter_name="mode",
                value_type="string",
                default_value="strict",
                meaning="The interpretation mode.",
            ),
            SemanticParameterDefinition(
                parameter_name="attempt_count",
                value_type="integer",
                required=True,
                meaning="The number of attempts represented.",
            ),
        ),
        description="Original editorial description.",
    )


def test_default_semantics_has_stable_golden_fingerprint():
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )

    assert proposition_semantics_fingerprint(semantics) == (
        "sha256-v1:"
        "8b4f03c8451f30aaa31811c0363485334eaa9f41ec96dd445a2230ce026dad6c"
    )


def test_fingerprint_excludes_description():
    semantics = make_semantics()
    revised_description = replace(
        semantics,
        description="Completely different explanatory wording.",
    )

    assert proposition_semantics_fingerprint(semantics) == (
        proposition_semantics_fingerprint(revised_description)
    )


def test_canonical_payload_does_not_contain_description():
    payload = canonical_proposition_semantics_payload(make_semantics())

    assert "description" not in payload
    assert "Original editorial description." not in canonical_proposition_semantics_json(
        make_semantics()
    )


def test_fingerprint_is_independent_of_definition_tuple_order():
    semantics = make_semantics()
    reordered = replace(
        semantics,
        roles=tuple(reversed(semantics.roles)),
        temporal_constraints=tuple(reversed(semantics.temporal_constraints)),
        parameters=tuple(reversed(semantics.parameters)),
    )

    assert proposition_semantics_fingerprint(semantics) == (
        proposition_semantics_fingerprint(reordered)
    )
    assert canonical_proposition_semantics_json(semantics) == (
        canonical_proposition_semantics_json(reordered)
    )


def test_fingerprint_normalizes_unicode_to_nfc():
    semantics = make_semantics()
    composed = "Café account"
    decomposed = "Cafe\u0301 account"
    first = replace(
        semantics,
        roles=(
            replace(semantics.roles[0], meaning=composed),
            *semantics.roles[1:],
        ),
    )
    second = replace(
        semantics,
        roles=(
            replace(semantics.roles[0], meaning=decomposed),
            *semantics.roles[1:],
        ),
    )

    assert proposition_semantics_fingerprint(first) == (
        proposition_semantics_fingerprint(second)
    )


@pytest.mark.parametrize(
    "changed",
    [
        lambda s: replace(
            s,
            roles=(
                replace(s.roles[0], meaning="A materially different account meaning."),
                *s.roles[1:],
            ),
        ),
        lambda s: replace(
            s,
            roles=(
                replace(s.roles[0], required=False),
                *s.roles[1:],
            ),
        ),
        lambda s: replace(
            s,
            temporal_constraints=(
                replace(s.temporal_constraints[0], max_interval_seconds=121),
            ),
        ),
        lambda s: replace(
            s,
            temporal_scope=replace(s.temporal_scope, end_inclusive=True),
        ),
        lambda s: replace(
            s,
            parameters=(
                replace(s.parameters[0], default_value="permissive"),
                s.parameters[1],
            ),
        ),
        lambda s: replace(
            s,
            parameters=(
                replace(s.parameters[0], meaning="A different interpretation mode."),
                s.parameters[1],
            ),
        ),
    ],
)
def test_meaning_bearing_definition_changes_change_fingerprint(changed):
    semantics = make_semantics()

    assert proposition_semantics_fingerprint(semantics) != (
        proposition_semantics_fingerprint(changed(semantics))
    )


def test_fingerprint_is_versioned_and_has_sha256_digest_shape():
    fingerprint = proposition_semantics_fingerprint(make_semantics())

    assert fingerprint.startswith(SEMANTICS_FINGERPRINT_ALGORITHM + ":")
    assert re.fullmatch(r"sha256-v1:[0-9a-f]{64}", fingerprint)


def test_canonical_payload_includes_all_definition_schema_versions():
    payload = canonical_proposition_semantics_payload(make_semantics())

    assert payload["schema_version"] == 1
    assert all(item["schema_version"] == 1 for item in payload["roles"])
    assert all(
        item["schema_version"] == 1
        for item in payload["temporal_constraints"]
    )
    assert payload["temporal_scope"]["schema_version"] == 1
    assert all(item["schema_version"] == 1 for item in payload["parameters"])


def test_fingerprint_rejects_non_semantics_input():
    with pytest.raises(TypeError, match="PropositionSemantics"):
        proposition_semantics_fingerprint("not semantics")



def test_fingerprint_rejects_role_names_colliding_after_unicode_normalization():
    semantics = PropositionSemantics(
        proposition_type="unicode_names",
        semantics_version=1,
        roles=(
            PropositionRoleDefinition(
                role_name="Café",
                role_kind="event",
                cardinality="one",
                event_type="authentication_failure",
                meaning="First event.",
            ),
            PropositionRoleDefinition(
                role_name="Cafe\u0301",
                role_kind="event",
                cardinality="one",
                event_type="authentication_success",
                meaning="Second event.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="collide after Unicode NFC"):
        proposition_semantics_fingerprint(semantics)
