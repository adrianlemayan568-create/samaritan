
from dataclasses import FrozenInstanceError

import pytest

from samaritan.proposition_semantics import (
    PropositionRoleDefinition,
    TemporalConstraintDefinition,
    TemporalScopeDefinition,
    SemanticParameterDefinition,
    PropositionSemantics,
    PropositionSemanticsRegistry,
    build_default_proposition_semantics_registry,
)


def make_role(
    name="event",
    kind="event",
    target="authentication_failure",
    meaning=None,
    required=True,
):
    if meaning is None:
        meaning = (
            "The entity represented by this role."
            if kind == "entity"
            else "The event represented by this role."
        )

    if kind == "entity":
        return PropositionRoleDefinition(
            role_name=name,
            role_kind="entity",
            cardinality="one",
            entity_type=target,
            required=required,
            meaning=meaning,
        )

    return PropositionRoleDefinition(
        role_name=name,
        role_kind="event",
        cardinality="one",
        event_type=target,
        required=required,
        meaning=meaning,
    )



def make_semantics(
    *,
    proposition_type="test_proposition",
    semantics_version=1,
    roles=None,
    temporal_constraints=(),
    temporal_scope=None,
    parameters=(),
):
    return PropositionSemantics(
        proposition_type=proposition_type,
        semantics_version=semantics_version,
        roles=roles if roles is not None else (make_role(),),
        temporal_constraints=temporal_constraints,
        temporal_scope=temporal_scope,
        parameters=parameters,
        description="A test proposition.",
    )


def test_default_registry_contains_initial_semantics():
    registry = build_default_proposition_semantics_registry()

    assert len(registry) == 1
    assert registry.contains("account_authentication_sequence", 1)


def test_initial_semantics_has_expected_roles():
    registry = build_default_proposition_semantics_registry()
    semantics = registry.get("account_authentication_sequence", 1)

    assert [(role.role_name, role.role_kind) for role in semantics.roles] == [
        ("account", "entity"),
        ("failure_event", "event"),
        ("success_event", "event"),
        ("host_activity_event", "event"),
    ]

    assert semantics.roles[0].entity_type == "user_account"
    assert semantics.roles[1].event_type == "authentication_failure"
    assert semantics.roles[2].event_type == "authentication_success"
    assert semantics.roles[3].event_type == "host_activity"


def test_initial_semantics_defines_role_meanings():
    registry = build_default_proposition_semantics_registry()
    semantics = registry.get("account_authentication_sequence", 1)

    assert [role.meaning for role in semantics.roles] == [
        "The user account associated with the authentication events.",
        "An authentication failure associated with the specified account.",
        "A subsequent successful authentication associated with that account.",
        "Subsequent host activity associated with that account.",
    ]


def test_initial_semantics_has_required_temporal_order():
    registry = build_default_proposition_semantics_registry()
    semantics = registry.get("account_authentication_sequence", 1)

    assert [
        (item.earlier_role, item.later_role, item.relation)
        for item in semantics.temporal_constraints
    ] == [
        ("failure_event", "success_event", "precedes"),
        ("success_event", "host_activity_event", "precedes"),
    ]


def test_initial_semantics_has_expected_temporal_scope():
    registry = build_default_proposition_semantics_registry()
    scope = registry.get("account_authentication_sequence", 1).temporal_scope

    assert scope is not None
    assert scope.start_role == "failure_event"
    assert scope.end_role == "host_activity_event"
    assert scope.start_inclusive is True
    assert scope.end_inclusive is True


def test_temporal_scope_definition_is_immutable():
    scope = TemporalScopeDefinition(
        start_role="failure",
        end_role="success",
    )

    with pytest.raises(FrozenInstanceError):
        scope.start_role = "changed"


def test_temporal_scope_accepts_transitive_temporal_path():
    roles = (
        make_role("first"),
        make_role("middle", target="authentication_success"),
        make_role("last", target="host_activity"),
    )

    semantics = make_semantics(
        roles=roles,
        temporal_constraints=(
            TemporalConstraintDefinition(
                earlier_role="first",
                later_role="middle",
            ),
            TemporalConstraintDefinition(
                earlier_role="middle",
                later_role="last",
            ),
        ),
        temporal_scope=TemporalScopeDefinition(
            start_role="first",
            end_role="last",
        ),
    )

    assert semantics.temporal_scope.start_role == "first"
    assert semantics.temporal_scope.end_role == "last"


def test_temporal_scope_rejects_identical_boundary_roles():
    with pytest.raises(ValueError):
        TemporalScopeDefinition(
            start_role="event",
            end_role="event",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_inclusive", 1),
        ("start_inclusive", None),
        ("end_inclusive", 0),
        ("end_inclusive", "true"),
    ],
)
def test_temporal_scope_rejects_non_boolean_inclusivity(field, value):
    kwargs = {
        "start_role": "failure",
        "end_role": "success",
        field: value,
    }

    with pytest.raises(TypeError):
        TemporalScopeDefinition(**kwargs)


def test_temporal_scope_rejects_entity_role_boundary():
    account = make_role(
        name="account",
        kind="entity",
        target="user_account",
    )
    event = make_role(name="event")

    with pytest.raises(ValueError, match="event roles"):
        make_semantics(
            roles=(account, event),
            temporal_scope=TemporalScopeDefinition(
                start_role="account",
                end_role="event",
            ),
        )


def test_temporal_scope_rejects_optional_boundary_role():
    roles = (
        make_role("first"),
        make_role("middle", target="authentication_success"),
        make_role("last", target="host_activity", required=False),
    )

    with pytest.raises(ValueError, match="required event roles"):
        make_semantics(
            roles=roles,
            temporal_scope=TemporalScopeDefinition(
                start_role="first",
                end_role="last",
            ),
        )



def test_temporal_scope_rejects_missing_temporal_path():
    roles = (
        make_role("first"),
        make_role("second", target="authentication_success"),
    )

    with pytest.raises(ValueError, match="must precede"):
        make_semantics(
            roles=roles,
            temporal_scope=TemporalScopeDefinition(
                start_role="first",
                end_role="second",
            ),
        )


def test_semantics_does_not_hardcode_proposition_specific_scope():
    roles = (
        make_role("account", kind="entity", target="user_account"),
        make_role("failure_event"),
        make_role("success_event", target="authentication_success"),
        make_role("host_activity_event", target="host_activity"),
    )

    semantics = make_semantics(
        proposition_type="account_authentication_sequence",
        semantics_version=1,
        roles=roles,
        temporal_constraints=(
            TemporalConstraintDefinition("failure_event", "success_event"),
            TemporalConstraintDefinition("success_event", "host_activity_event"),
        ),
    )

    assert semantics.temporal_scope is None



def test_semantics_definition_is_immutable():
    semantics = make_semantics()

    with pytest.raises(FrozenInstanceError):
        semantics.proposition_type = "changed"


def test_role_definition_is_immutable():
    role = make_role()

    with pytest.raises(FrozenInstanceError):
        role.role_name = "changed"


def test_role_definition_requires_meaning_argument():
    with pytest.raises(TypeError, match="meaning"):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="one",
            event_type="authentication_failure",
        )



def test_role_definition_rejects_blank_meaning():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="one",
            event_type="authentication_failure",
            meaning="   ",
        )


def test_role_definition_rejects_non_string_meaning():
    with pytest.raises(TypeError):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="one",
            event_type="authentication_failure",
            meaning=123,
        )


def test_entity_role_requires_entity_type():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="account",
            role_kind="entity",
            cardinality="one",
            meaning="A test role.",
        )



def test_entity_role_rejects_event_type():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="account",
            role_kind="entity",
            cardinality="one",
            entity_type="user_account",
            event_type="authentication_failure",
            meaning="A test role.",
        )



def test_event_role_requires_event_type():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="one",
            meaning="A test role.",
        )



def test_event_role_rejects_entity_type():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="one",
            event_type="authentication_failure",
            entity_type="user_account",
            meaning="A test role.",
        )



def test_unsupported_role_kind_is_rejected():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="role",
            role_kind="relationship",
            cardinality="one",
            event_type="authentication_failure",
            meaning="A test role.",
        )



def test_unsupported_cardinality_is_rejected():
    with pytest.raises(ValueError):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="many",
            event_type="authentication_failure",
            meaning="A test role.",
        )



def test_required_must_be_boolean():
    with pytest.raises(TypeError):
        PropositionRoleDefinition(
            role_name="event",
            role_kind="event",
            cardinality="one",
            event_type="authentication_failure",
            required=1,
            meaning="A test role.",
        )



def test_duplicate_role_names_are_rejected():
    role = make_role()

    with pytest.raises(ValueError, match="unique"):
        make_semantics(roles=(role, role))


def test_roles_must_be_nonempty_tuple():
    with pytest.raises(TypeError):
        make_semantics(roles=[])


def test_roles_must_contain_role_definitions():
    with pytest.raises(TypeError):
        make_semantics(roles=("event",))


def test_temporal_constraint_rejects_same_role():
    with pytest.raises(ValueError):
        TemporalConstraintDefinition(
            earlier_role="event",
            later_role="event",
        )


def test_unsupported_temporal_relation_is_rejected():
    with pytest.raises(ValueError):
        TemporalConstraintDefinition(
            earlier_role="first",
            later_role="second",
            relation="causes",
        )


def test_temporal_constraint_must_reference_event_roles():
    account = make_role(
        name="account",
        kind="entity",
        target="user_account",
    )
    event = make_role(name="event")

    with pytest.raises(ValueError, match="event roles"):
        make_semantics(
            roles=(account, event),
            temporal_constraints=(
                TemporalConstraintDefinition(
                    earlier_role="account",
                    later_role="event",
                ),
            ),
        )


def test_temporal_constraint_rejects_optional_event_role():
    roles = (
        make_role("required"),
        make_role("optional", target="authentication_success", required=False),
    )

    with pytest.raises(ValueError, match="required event roles"):
        make_semantics(
            roles=roles,
            temporal_constraints=(
                TemporalConstraintDefinition("required", "optional"),
            ),
        )


def test_temporal_constraint_rejects_unknown_role():
    with pytest.raises(ValueError, match="event roles"):
        make_semantics(
            temporal_constraints=(
                TemporalConstraintDefinition(
                    earlier_role="unknown",
                    later_role="event",
                ),
            ),
        )


def test_duplicate_temporal_constraints_are_rejected():
    constraint = TemporalConstraintDefinition(
        earlier_role="first",
        later_role="second",
    )

    with pytest.raises(ValueError, match="duplicate"):
        make_semantics(
            roles=(
                make_role("first"),
                make_role("second", target="authentication_success"),
            ),
            temporal_constraints=(constraint, constraint),
        )


def test_temporal_constraints_must_be_tuple():
    with pytest.raises(TypeError):
        make_semantics(temporal_constraints=[])


def test_temporal_constraints_must_have_correct_types():
    with pytest.raises(TypeError):
        make_semantics(temporal_constraints=("constraint",))


@pytest.mark.parametrize("value", [True, 1.5, "1", None])
def test_semantics_version_rejects_non_integer_types(value):
    with pytest.raises(TypeError, match="must be an integer"):
        make_semantics(semantics_version=value)


@pytest.mark.parametrize("value", [0, -1])
def test_semantics_version_must_be_positive(value):
    with pytest.raises(ValueError, match="positive integer"):
        make_semantics(semantics_version=value)



def test_parameter_definition_accepts_string_default():
    parameter = SemanticParameterDefinition(
        parameter_name="mode",
        value_type="string",
        default_value="strict",
    )

    assert parameter.default_value == "strict"


def test_parameter_definition_accepts_integer_default():
    parameter = SemanticParameterDefinition(
        parameter_name="limit",
        value_type="integer",
        default_value=5,
    )

    assert parameter.default_value == 5


def test_parameter_definition_accepts_boolean_default():
    parameter = SemanticParameterDefinition(
        parameter_name="enabled",
        value_type="boolean",
        default_value=True,
    )

    assert parameter.default_value is True


def test_required_parameter_cannot_have_default():
    with pytest.raises(ValueError):
        SemanticParameterDefinition(
            parameter_name="mode",
            value_type="string",
            required=True,
            default_value="strict",
        )


def test_integer_parameter_rejects_boolean_default():
    with pytest.raises(TypeError):
        SemanticParameterDefinition(
            parameter_name="limit",
            value_type="integer",
            default_value=True,
        )


def test_boolean_parameter_rejects_integer_default():
    with pytest.raises(TypeError):
        SemanticParameterDefinition(
            parameter_name="enabled",
            value_type="boolean",
            default_value=1,
        )


def test_string_parameter_rejects_non_string_default():
    with pytest.raises(TypeError):
        SemanticParameterDefinition(
            parameter_name="mode",
            value_type="string",
            default_value=1,
        )


def test_string_parameter_rejects_empty_default():
    with pytest.raises(ValueError):
        SemanticParameterDefinition(
            parameter_name="mode",
            value_type="string",
            default_value="",
        )


def test_unsupported_parameter_type_is_rejected():
    with pytest.raises(ValueError):
        SemanticParameterDefinition(
            parameter_name="value",
            value_type="object",
        )


@pytest.mark.parametrize("value_type", ["decimal", "entity_id", "object"])
def test_unsupported_parameter_types_are_rejected(value_type):
    with pytest.raises(ValueError, match="unsupported semantic parameter type"):
        SemanticParameterDefinition(
            parameter_name="value",
            value_type=value_type,
        )


def test_duplicate_parameter_names_are_rejected():
    parameter = SemanticParameterDefinition(
        parameter_name="mode",
        value_type="string",
        default_value="strict",
    )

    with pytest.raises(ValueError, match="unique"):
        make_semantics(parameters=(parameter, parameter))


def test_parameters_must_be_tuple():
    with pytest.raises(TypeError):
        make_semantics(parameters=[])


def test_parameters_must_contain_parameter_definitions():
    with pytest.raises(TypeError):
        make_semantics(parameters=("parameter",))


def test_semantics_description_may_be_empty():
    semantics = PropositionSemantics(
        proposition_type="test",
        semantics_version=1,
        roles=(make_role(),),
    )

    assert semantics.description == ""


def test_semantics_rejects_surrounding_whitespace():
    with pytest.raises(ValueError):
        make_semantics(proposition_type=" test ")


def test_registry_rejects_non_semantics_definition():
    registry = PropositionSemanticsRegistry()

    with pytest.raises(TypeError):
        registry.register("invalid")


def test_registry_rejects_duplicate_registration():
    registry = PropositionSemanticsRegistry()
    semantics = make_semantics()

    registry.register(semantics)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(semantics)


def test_registry_retrieves_exact_version():
    registry = PropositionSemanticsRegistry()
    version_one = make_semantics(semantics_version=1)
    version_two = make_semantics(semantics_version=2)

    registry.register(version_one)
    registry.register(version_two)

    assert registry.get("test_proposition", 1) is version_one
    assert registry.get("test_proposition", 2) is version_two


def test_registry_does_not_fall_back_to_another_version():
    registry = PropositionSemanticsRegistry()
    registry.register(make_semantics(semantics_version=1))

    with pytest.raises(KeyError, match="test_proposition@2"):
        registry.get("test_proposition", 2)


@pytest.mark.parametrize("version", [True, 1.5, "1", None])
def test_registry_rejects_non_integer_version_lookup(version):
    registry = PropositionSemanticsRegistry()

    with pytest.raises(TypeError, match="must be an integer"):
        registry.get("test_proposition", version)


@pytest.mark.parametrize("version", [0, -1])
def test_registry_rejects_non_positive_version_lookup(version):
    registry = PropositionSemanticsRegistry()

    with pytest.raises(ValueError, match="positive integer"):
        registry.get("test_proposition", version)



def test_registry_contains_returns_false_for_unknown_version():
    registry = PropositionSemanticsRegistry()
    registry.register(make_semantics())

    assert registry.contains("test_proposition", 1)
    assert not registry.contains("test_proposition", 2)


def test_registry_returns_the_registered_semantics_definition():
    registry = PropositionSemanticsRegistry()
    semantics = make_semantics()

    registry.register(semantics)

    assert registry.get("test_proposition", 1) is semantics
    assert len(registry) == 1


def test_two_role_temporal_cycle_is_rejected():
    with pytest.raises(ValueError, match="must not contain cycles"):
        make_semantics(
            roles=(make_role("first"), make_role("second")),
            temporal_constraints=(
                TemporalConstraintDefinition("first", "second"),
                TemporalConstraintDefinition("second", "first"),
            ),
        )


def test_three_role_temporal_cycle_is_rejected():
    with pytest.raises(ValueError, match="must not contain cycles"):
        make_semantics(
            roles=(
                make_role("first"),
                make_role("second"),
                make_role("third"),
            ),
            temporal_constraints=(
                TemporalConstraintDefinition("first", "second"),
                TemporalConstraintDefinition("second", "third"),
                TemporalConstraintDefinition("third", "first"),
            ),
        )


def test_acyclic_temporal_chain_is_accepted():
    semantics = make_semantics(
        roles=(
            make_role("first"),
            make_role("second"),
            make_role("third"),
        ),
        temporal_constraints=(
            TemporalConstraintDefinition("first", "second"),
            TemporalConstraintDefinition("second", "third"),
        ),
    )

    assert len(semantics.temporal_constraints) == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PropositionRoleDefinition(
            "event",
            "event",
            "one", "A test role.",
            event_type="authentication_failure",
            schema_version=True,
        ),
        lambda: TemporalConstraintDefinition(
            "first", "second", schema_version=True,
        ),
        lambda: SemanticParameterDefinition(
            "mode", "string", default_value="strict", schema_version=True,
        ),
        lambda: PropositionSemantics(
            "test", 1, (make_role(),), schema_version=True,
        ),
    ],
)
def test_schema_versions_reject_boolean_values(factory):
    with pytest.raises(TypeError, match="must be an integer"):
        factory()


@pytest.mark.parametrize("description", [None, 0, False, 123])
def test_semantics_description_rejects_non_strings(description):
    with pytest.raises(TypeError, match="description must be a string"):
        PropositionSemantics(
            proposition_type="test",
            semantics_version=1,
            roles=(make_role(),),
            description=description,
        )


def test_semantics_description_rejects_surrounding_whitespace():
    with pytest.raises(ValueError):
        PropositionSemantics(
            proposition_type="test",
            semantics_version=1,
            roles=(make_role(),),
            description=" invalid ",
        )


def test_default_authentication_sequence_limits_each_temporal_gap_to_five_minutes():
    semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    assert [item.max_interval_seconds for item in semantics.temporal_constraints] == [
        300,
        300,
    ]


def test_temporal_constraint_rejects_non_positive_interval_limit():
    with pytest.raises(ValueError, match="positive"):
        TemporalConstraintDefinition(
            "first",
            "second",
            max_interval_seconds=0,
        )


def test_temporal_constraint_requires_integer_interval_limit():
    with pytest.raises(TypeError, match="integer"):
        TemporalConstraintDefinition(
            "first",
            "second",
            max_interval_seconds=1.5,
        )


@pytest.mark.parametrize(
    "identifier",
    ["bad\\nname", "bad\\x00name", "name@version", "a" * 129, "аccount"],
)
def test_semantics_rejects_hostile_or_ambiguous_identifiers(identifier):
    with pytest.raises(ValueError):
        make_semantics(proposition_type=identifier)


def test_semantics_registry_can_be_frozen_and_exposes_read_only_definitions():
    registry = PropositionSemanticsRegistry()
    definition = make_semantics()
    registry.register(definition)
    registry.freeze()

    assert registry.get(definition.proposition_type, definition.semantics_version) is definition
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(make_semantics(semantics_version=2))
    with pytest.raises(TypeError):
        registry.definitions[(definition.proposition_type, 2)] = make_semantics(semantics_version=2)


def test_semantics_rejects_unvalidated_role_definition_subclasses():
    class ExtendedRoleDefinition(PropositionRoleDefinition):
        pass

    role = ExtendedRoleDefinition(
        role_name="event",
        role_kind="event",
        cardinality="one",
        event_type="authentication_failure",
        meaning="A test role.",
    )
    with pytest.raises(TypeError, match="PropositionRoleDefinition"):
        make_semantics(roles=(role,))
