from datetime import datetime, timedelta, timezone

import pytest

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import (
    Entity,
    Event,
    EventParticipant,
    Relationship,
    Observation,
    Source,
    new_id,
    payload_checksum,
)
from samaritan.database import Database
from samaritan.evaluation_contracts import EvidenceSet
from samaritan.proposition_construction import (
    PropositionConstructionError,
    PropositionConstructor,
)
from samaritan.proposition_construction_contracts import (
    EventAttributeParameterBinding,
    EntityRoleBinding,
    TemporalDifferenceParameterBinding,
    PropositionConstructionDefinition,
    ParticipantRoleBinding,
    RelationshipParameterBinding,
    PropositionConstructionRegistry,
    build_default_proposition_construction_registry,
)
from samaritan.proposition_semantics import (
    PropositionRoleDefinition,
    PropositionSemantics,
    PropositionSemanticsRegistry,
    SemanticParameterDefinition,
    build_default_proposition_semantics_registry,
)
from samaritan.repository import Repository


def make_repository(tmp_path):
    database = Database(tmp_path / "construction.db")
    database.initialize()
    repository = Repository(database)
    now = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    source = Source(
        source_id=new_id(),
        source_type="synthetic",
        name="proposition construction tests",
        created_at=now,
    )
    repository.register_source(source)
    return repository, source, now


def persist_event(repository, source, entity, event_type, role, occurred_at, normalized_data_json='{}'):
    observation_id = new_id()
    observation = Observation(
        observation_id=observation_id,
        source_id=source.source_id,
        received_at=occurred_at,
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    event = Event(
        event_id=new_id(),
        observation_id=observation_id,
        event_type=event_type,
        occurred_at=occurred_at,
        original_timestamp=occurred_at.isoformat().replace("+00:00", "Z"),
        normalized_data_json=normalized_data_json,
        participants=(EventParticipant(entity.entity_id, role),),
    )
    entities = (entity,) if repository.get_entities_by_ids((entity.entity_id,)) == () else ()
    repository.persist_observation(observation, entities=entities, events=(event,))
    return event


def make_constructor(repository):
    return PropositionConstructor(
        repository,
        build_default_proposition_semantics_registry(),
        build_default_proposition_construction_registry(),
    )


def test_constructs_account_authentication_sequence_from_event_references(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(repository, source, account, "authentication_failure", "target", start),
        persist_event(repository, source, account, "authentication_success", "target", start + timedelta(seconds=90)),
        persist_event(repository, source, account, "host_activity", "subject", start + timedelta(seconds=180)),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=4),
    )
    proposition = make_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )
    roles = {item.role_name: item for item in proposition.roles}
    assert roles["account"].value_id == account.entity_id
    assert roles["failure_event"].value_id == events[0].event_id
    assert roles["success_event"].value_id == events[1].event_id
    assert roles["host_activity_event"].value_id == events[2].event_id
    account_provenance = {
        item.field_name: item for item in proposition.field_provenance
    }["account"]
    assert account_provenance.event_ids == tuple(item.event_id for item in events)
    assert len(account_provenance.observation_ids) == 3


def test_constructor_rejects_temporal_gap_over_five_minutes(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(repository, source, account, "authentication_failure", "target", start),
        persist_event(repository, source, account, "authentication_success", "target", start + timedelta(seconds=301)),
        persist_event(repository, source, account, "host_activity", "subject", start + timedelta(seconds=360)),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=7),
    )
    with pytest.raises(PropositionConstructionError, match="exceeds 300 seconds"):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_ambiguous_multiple_events_for_one_semantic_role(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(repository, source, account, "authentication_failure", "target", start),
        persist_event(repository, source, account, "authentication_failure", "target", start + timedelta(seconds=10)),
        persist_event(repository, source, account, "authentication_success", "target", start + timedelta(seconds=90)),
        persist_event(repository, source, account, "host_activity", "subject", start + timedelta(seconds=180)),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=4),
    )
    with pytest.raises(PropositionConstructionError, match="found 2"):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )



def make_parameter_constructor(repository, parameter, *, event_attribute_bindings=(), temporal_parameter_bindings=()):
    base_semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(
        PropositionSemantics(
            proposition_type=base_semantics.proposition_type,
            semantics_version=base_semantics.semantics_version,
            roles=base_semantics.roles,
            temporal_constraints=base_semantics.temporal_constraints,
            parameters=(parameter,),
            description=base_semantics.description,
            temporal_scope=base_semantics.temporal_scope,
        )
    )
    base_construction = build_default_proposition_construction_registry().get(
        "account_authentication_sequence", 1
    )
    construction = PropositionConstructionDefinition(
        construction_id=base_construction.construction_id,
        construction_version=base_construction.construction_version,
        proposition_type=base_construction.proposition_type,
        semantics_version=base_construction.semantics_version,
        event_bindings=base_construction.event_bindings,
        entity_bindings=base_construction.entity_bindings,
        description=base_construction.description,
        event_attribute_bindings=event_attribute_bindings,
        temporal_parameter_bindings=temporal_parameter_bindings,
    )
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(construction)
    return PropositionConstructor(
        repository,
        semantics_registry,
        construction_registry,
    )


def test_event_attribute_parameter_binding_preserves_source_provenance(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(repository, source, account, "authentication_failure", "target", start),
        persist_event(
            repository, source, account, "authentication_success", "target",
            start + timedelta(seconds=90), '{"result":"success"}',
        ),
        persist_event(repository, source, account, "host_activity", "subject", start + timedelta(seconds=180)),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=4),
    )
    parameter = SemanticParameterDefinition(
        "authentication_result",
        "string",
        required=True,
        meaning="The normalized result of the successful authentication event.",
    )
    constructor = make_parameter_constructor(
        repository,
        parameter,
        event_attribute_bindings=(
            EventAttributeParameterBinding(
                "authentication_result",
                "success_event",
                "result",
                "Read the registered normalized result attribute.",
            ),
        ),
    )
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    assert proposition.parameters_json == '{"authentication_result":"success"}'
    provenance = {
        item.field_name: item for item in proposition.field_provenance
    }["authentication_result"]
    assert provenance.event_ids == (events[1].event_id,)
    assert provenance.observation_ids == (events[1].observation_id,)


def test_temporal_difference_parameter_binding_is_deterministic(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(repository, source, account, "authentication_failure", "target", start),
        persist_event(repository, source, account, "authentication_success", "target", start + timedelta(seconds=90)),
        persist_event(repository, source, account, "host_activity", "subject", start + timedelta(seconds=180)),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=4),
    )
    parameter = SemanticParameterDefinition(
        "failure_to_success_seconds",
        "integer",
        required=True,
        meaning="Elapsed whole seconds from failure to success.",
    )
    constructor = make_parameter_constructor(
        repository,
        parameter,
        temporal_parameter_bindings=(
            TemporalDifferenceParameterBinding(
                "failure_to_success_seconds",
                "failure_event",
                "success_event",
                "Derive elapsed seconds from the validated event timestamps.",
            ),
        ),
    )
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    assert proposition.parameters_json == '{"failure_to_success_seconds":90}'
    provenance = {
        item.field_name: item for item in proposition.field_provenance
    }["failure_to_success_seconds"]
    assert provenance.event_ids == (events[0].event_id, events[1].event_id)



@pytest.mark.parametrize("duplicate_relationship", [False, True])
def test_relationship_parameter_binding_requires_and_preserves_explicit_relationship(
    tmp_path, duplicate_relationship
):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    device = Entity(
        entity_id=new_id(),
        entity_type="host",
        canonical_identifier="workstation-01",
        created_at=start,
    )
    failure = persist_event(
        repository, source, account, "authentication_failure", "target", start
    )
    success = persist_event(
        repository, source, account, "authentication_success", "target",
        start + timedelta(seconds=90),
    )

    host_time = start + timedelta(seconds=180)
    observation = Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=host_time,
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    host_event = Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type="host_activity",
        occurred_at=host_time,
        original_timestamp=host_time.isoformat().replace("+00:00", "Z"),
        normalized_data_json="{}",
        participants=(
            EventParticipant(account.entity_id, "subject"),
            EventParticipant(device.entity_id, "device"),
        ),
    )
    relationship = Relationship(
        relationship_id=new_id(),
        source_entity_id=account.entity_id,
        target_entity_id=device.entity_id,
        relationship_type="account_uses_device",
        observation_id=observation.observation_id,
        observed_at=host_time,
    )
    relationships = (relationship,)
    if duplicate_relationship:
        relationships += (
            Relationship(
                relationship_id=new_id(),
                source_entity_id=account.entity_id,
                target_entity_id=device.entity_id,
                relationship_type="account_uses_device",
                observation_id=observation.observation_id,
                observed_at=host_time,
            ),
        )

    repository.persist_observation(
        observation,
        entities=(device,),
        events=(host_event,),
        relationships=relationships,
    )

    base_semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    device_role = PropositionRoleDefinition(
        role_name="device",
        role_kind="entity",
        cardinality="one",
        entity_type="host",
        meaning="The host device associated with the host activity.",
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(
        PropositionSemantics(
            proposition_type=base_semantics.proposition_type,
            semantics_version=base_semantics.semantics_version,
            roles=base_semantics.roles + (device_role,),
            temporal_constraints=base_semantics.temporal_constraints,
            parameters=(
                SemanticParameterDefinition(
                    "account_device_relationship_id",
                    "string",
                    required=True,
                    meaning="The explicit relationship record connecting the account to the device.",
                ),
            ),
            description=base_semantics.description,
            temporal_scope=base_semantics.temporal_scope,
        )
    )
    base_construction = build_default_proposition_construction_registry().get(
        "account_authentication_sequence", 1
    )
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(
        PropositionConstructionDefinition(
            construction_id=base_construction.construction_id,
            construction_version=base_construction.construction_version,
            proposition_type=base_construction.proposition_type,
            semantics_version=base_construction.semantics_version,
            event_bindings=base_construction.event_bindings,
            entity_bindings=base_construction.entity_bindings + (
                EntityRoleBinding(
                    "device",
                    (ParticipantRoleBinding("host_activity_event", "device"),),
                ),
            ),
            description=base_construction.description,
            relationship_parameter_bindings=(
                RelationshipParameterBinding(
                    "account_device_relationship_id",
                    "account",
                    "device",
                    "account_uses_device",
                    "relationship_id",
                    "Use the explicit directed account-to-device relationship record.",
                ),
            ),
        )
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(
            EvidenceReference(event_id=item.event_id)
            for item in (failure, success, host_event)
        ),
        created_at=host_time + timedelta(minutes=1),
    )
    constructor = PropositionConstructor(
        repository,
        semantics_registry,
        construction_registry,
    )

    if duplicate_relationship:
        with pytest.raises(
            PropositionConstructionError,
            match="expected exactly one relationship of type account_uses_device; found 2",
        ):
            constructor.construct(
                evidence_set, "account_authentication_sequence", 1
            )
        return

    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )

    assert {
        item.role_name: item.value_id for item in proposition.roles
    }["device"] == device.entity_id
    assert proposition.parameters_json == (
        '{"account_device_relationship_id":"' + relationship.relationship_id + '"}'
    )
    provenance = {
        item.field_name: item for item in proposition.field_provenance
    }["account_device_relationship_id"]
    assert provenance.relationship_ids == (relationship.relationship_id,)
    assert provenance.observation_ids == (observation.observation_id,)


def _make_authentication_evidence(
    repository,
    source,
    account,
    start,
    *,
    failure_seconds=0,
    success_seconds=90,
    host_seconds=180,
):
    events = (
        persist_event(
            repository, source, account, "authentication_failure", "target",
            start + timedelta(seconds=failure_seconds),
        ),
        persist_event(
            repository, source, account, "authentication_success", "target",
            start + timedelta(seconds=success_seconds),
        ),
        persist_event(
            repository, source, account, "host_activity", "subject",
            start + timedelta(seconds=host_seconds),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(
            EvidenceReference(event_id=item.event_id) for item in events
        ),
        created_at=start + timedelta(seconds=max(
            failure_seconds, success_seconds, host_seconds
        ) + 60),
    )
    return events, evidence_set


def test_constructor_rejects_events_resolving_to_different_accounts(tmp_path):
    repository, source, start = make_repository(tmp_path)
    first_account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    second_account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="bob",
        created_at=start,
    )
    failure = persist_event(
        repository, source, first_account, "authentication_failure", "target", start
    )
    success = persist_event(
        repository, source, second_account, "authentication_success", "target",
        start + timedelta(seconds=90),
    )
    host = persist_event(
        repository, source, first_account, "host_activity", "subject",
        start + timedelta(seconds=180),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(
            EvidenceReference(event_id=item.event_id)
            for item in (failure, success, host)
        ),
        created_at=start + timedelta(minutes=4),
    )

    with pytest.raises(
        PropositionConstructionError,
        match="resolves to different entities across source events",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_missing_required_event_role(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(
            repository, source, account, "authentication_failure", "target", start
        ),
        persist_event(
            repository, source, account, "authentication_success", "target",
            start + timedelta(seconds=90),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=3),
    )

    with pytest.raises(
        PropositionConstructionError,
        match="expected exactly one host_activity event for host_activity_event; found 0",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_multiple_participants_for_semantic_role(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    other_account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="bob",
        created_at=start,
    )

    failure_time = start
    observation = Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=failure_time,
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    failure = Event(
        event_id=new_id(),
        observation_id=observation.observation_id,
        event_type="authentication_failure",
        occurred_at=failure_time,
        original_timestamp=failure_time.isoformat().replace("+00:00", "Z"),
        normalized_data_json="{}",
        participants=(
            EventParticipant(account.entity_id, "target"),
            EventParticipant(other_account.entity_id, "target"),
        ),
    )
    repository.persist_observation(
        observation,
        entities=(account, other_account),
        events=(failure,),
    )
    success = persist_event(
        repository, source, account, "authentication_success", "target",
        start + timedelta(seconds=90),
    )
    host = persist_event(
        repository, source, account, "host_activity", "subject",
        start + timedelta(seconds=180),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(
            EvidenceReference(event_id=item.event_id)
            for item in (failure, success, host)
        ),
        created_at=start + timedelta(minutes=4),
    )

    with pytest.raises(
        PropositionConstructionError,
        match="expected exactly one participant with role 'target' in failure_event; found 2",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_failure_event_after_success_event(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start,
        failure_seconds=90, success_seconds=0, host_seconds=180,
    )

    with pytest.raises(
        ValueError,
        match="failure_event must precede success_event",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_equal_failure_and_success_timestamps(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start,
        failure_seconds=0, success_seconds=0, host_seconds=180,
    )

    with pytest.raises(
        ValueError,
        match="failure_event must precede success_event",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_host_activity_at_success_timestamp(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start,
        failure_seconds=0, success_seconds=90, host_seconds=90,
    )

    with pytest.raises(
        ValueError,
        match="success_event must precede host_activity_event",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_accepts_temporal_intervals_of_exactly_300_seconds(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events, evidence_set = _make_authentication_evidence(
        repository, source, account, start,
        failure_seconds=0, success_seconds=300, host_seconds=600,
    )

    proposition = make_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )

    roles = {item.role_name: item for item in proposition.roles}
    assert roles["failure_event"].occurred_at == events[0].occurred_at
    assert roles["success_event"].occurred_at == events[1].occurred_at
    assert roles["host_activity_event"].occurred_at == events[2].occurred_at
    assert proposition.temporal_scope.starts_at == events[0].occurred_at
    assert proposition.temporal_scope.ends_at == events[2].occurred_at


def test_constructor_rejects_success_to_host_interval_over_300_seconds(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start,
        failure_seconds=0, success_seconds=90, host_seconds=391,
    )

    with pytest.raises(
        ValueError,
        match="temporal interval exceeds 300 seconds",
    ):
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_missing_registered_event_attribute(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    parameter = SemanticParameterDefinition(
        "authentication_result",
        "string",
        required=True,
        meaning="The normalized result of the successful authentication event.",
    )
    constructor = make_parameter_constructor(
        repository,
        parameter,
        event_attribute_bindings=(
            EventAttributeParameterBinding(
                "authentication_result",
                "success_event",
                "result",
                "Read the registered normalized result attribute.",
            ),
        ),
    )

    with pytest.raises(
        PropositionConstructionError,
        match="registered event attribute is missing: result",
    ):
        constructor.construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_rejects_event_attribute_type_mismatch(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="alice",
        created_at=start,
    )
    events = (
        persist_event(
            repository, source, account,
            "authentication_failure", "target", start,
        ),
        persist_event(
            repository, source, account,
            "authentication_success", "target",
            start + timedelta(seconds=90),
            '{"result":200}',
        ),
        persist_event(
            repository, source, account,
            "host_activity", "subject",
            start + timedelta(seconds=180),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(
            EvidenceReference(event_id=item.event_id) for item in events
        ),
        created_at=start + timedelta(minutes=4),
    )
    parameter = SemanticParameterDefinition(
        "authentication_result",
        "string",
        required=True,
        meaning="The normalized result of the successful authentication event.",
    )
    constructor = make_parameter_constructor(
        repository,
        parameter,
        event_attribute_bindings=(
            EventAttributeParameterBinding(
                "authentication_result",
                "success_event",
                "result",
                "Read the registered normalized result attribute.",
            ),
        ),
    )

    with pytest.raises(
        PropositionConstructionError,
        match="event attribute type mismatch for parameter authentication_result",
    ):
        constructor.construct(
            evidence_set, "account_authentication_sequence", 1
        )

