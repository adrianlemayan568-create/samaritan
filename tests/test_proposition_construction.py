import sqlite3
from dataclasses import replace
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
    EventRoleBinding,
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
from samaritan.proposition_contracts import (
    PropositionFieldProvenance,
    PropositionRoleValue,
    deterministic_proposition_id,
)
from samaritan.proposition_validation import (
    PropositionProvenanceError,
    PropositionProvenanceValidator,
)


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
@pytest.mark.parametrize("include_relationship_observation_reference", [False, True])
def test_relationship_parameter_binding_requires_selected_observation_and_preserves_explicit_relationship(
    tmp_path, duplicate_relationship, include_relationship_observation_reference
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
    evidence_references = [
        EvidenceReference(event_id=item.event_id)
        for item in (failure, success, host_event)
    ]
    if include_relationship_observation_reference:
        evidence_references.insert(
            2,
            EvidenceReference(observation_id=observation.observation_id),
        )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(evidence_references),
        created_at=host_time + timedelta(minutes=1),
    )
    constructor = PropositionConstructor(
        repository,
        semantics_registry,
        construction_registry,
    )

    if not include_relationship_observation_reference:
        with pytest.raises(
            PropositionConstructionError,
            match="expected exactly one relationship of type account_uses_device; found 0",
        ):
            constructor.construct(
                evidence_set, "account_authentication_sequence", 1
            )
        return

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

    event_only_manifest = tuple(
        item for item in proposition.evidence_manifest.evidence
        if item.observation_id is None
    )
    forged = replace(
        proposition,
        evidence_manifest=replace(
            proposition.evidence_manifest,
            evidence=event_only_manifest,
        ),
    )
    with pytest.raises(
        PropositionProvenanceError,
        match="expected exactly one relationship of type account_uses_device; found 0",
    ):
        PropositionProvenanceValidator(
            repository,
            semantics_registry,
            construction_registry,
        ).validate(forged)
    assert proposition.parameters_json == (
        '{"account_device_relationship_id":"' + relationship.relationship_id + '"}'
    )
    provenance = {
        item.field_name: item for item in proposition.field_provenance
    }["account_device_relationship_id"]
    assert provenance.relationship_ids == (relationship.relationship_id,)
    assert provenance.observation_ids == (observation.observation_id,)

    second_observation = Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=host_time + timedelta(seconds=10),
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    second_relationship = Relationship(
        relationship_id=new_id(),
        source_entity_id=account.entity_id,
        target_entity_id=device.entity_id,
        relationship_type="account_uses_device",
        observation_id=second_observation.observation_id,
        observed_at=host_time + timedelta(seconds=10),
    )
    repository.persist_observation(
        second_observation, relationships=(second_relationship,)
    )
    expanded_manifest = replace(
        proposition,
        evidence_manifest=replace(
            proposition.evidence_manifest,
            evidence=proposition.evidence_manifest.evidence
            + (EvidenceReference(observation_id=second_observation.observation_id),),
        ),
    )
    with pytest.raises(PropositionProvenanceError, match="expected exactly one relationship of type account_uses_device; found 2"):
        PropositionProvenanceValidator(
            repository, semantics_registry, construction_registry
        ).validate(expanded_manifest)


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

def test_provenance_validator_rejects_forged_event_observation_link(tmp_path):
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
    constructor = make_constructor(repository)
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    forged_provenance = tuple(
        replace(item, observation_ids=(new_id(),))
        if item.field_name == "failure_event"
        else item
        for item in proposition.field_provenance
    )
    forged = replace(proposition, field_provenance=forged_provenance)

    validator = PropositionProvenanceValidator(
        repository,
        constructor.semantics_registry,
        constructor.construction_registry,
    )
    with pytest.raises(
        PropositionProvenanceError,
        match="event and observation provenance disagree",
    ):
        validator.validate(forged)



def test_entity_role_provenance_allows_multiple_events_from_one_observation(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="shared-observation-account",
        created_at=start,
    )
    host_time = start + timedelta(seconds=180)
    observation = Observation(
        observation_id=new_id(),
        source_id=source.source_id,
        received_at=host_time,
        raw_payload="{}",
        payload_sha256=payload_checksum("{}"),
    )
    event_specs = (
        ("authentication_failure", "target", start),
        ("authentication_success", "target", start + timedelta(seconds=90)),
        ("host_activity", "subject", host_time),
    )
    events = tuple(
        Event(
            event_id=new_id(),
            observation_id=observation.observation_id,
            event_type=event_type,
            occurred_at=occurred_at,
            original_timestamp=occurred_at.isoformat().replace("+00:00", "Z"),
            normalized_data_json="{}",
            participants=(EventParticipant(account.entity_id, participant_role),),
        )
        for event_type, participant_role, occurred_at in event_specs
    )
    repository.persist_observation(
        observation,
        entities=(account,),
        events=events,
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=host_time + timedelta(minutes=1),
    )

    proposition = make_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )
    account_provenance = {
        item.field_name: item for item in proposition.field_provenance
    }["account"]
    assert account_provenance.event_ids == tuple(item.event_id for item in events)
    assert account_provenance.observation_ids == (observation.observation_id,)


def test_provenance_validator_rejects_populated_optional_role_without_binding(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier="optional-role-account",
        created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    base_semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    optional_role = PropositionRoleDefinition(
        role_name="optional_event",
        role_kind="event",
        cardinality="one",
        event_type="unregistered_optional_event",
        meaning="An optional event not supported by this construction.",
        required=False,
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(
        replace(base_semantics, roles=base_semantics.roles + (optional_role,))
    )
    construction_registry = build_default_proposition_construction_registry()
    constructor = PropositionConstructor(
        repository, semantics_registry, construction_registry
    )
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )

    optional_event = persist_event(
        repository,
        source,
        account,
        "unregistered_optional_event",
        "observer",
        start + timedelta(seconds=240),
    )
    optional_value = PropositionRoleValue(
        "optional_event",
        "event",
        optional_event.event_id,
        optional_event.occurred_at,
    )
    optional_provenance = PropositionFieldProvenance(
        "optional_event",
        (optional_event.event_id,),
        (optional_event.observation_id,),
    )
    roles = proposition.roles + (optional_value,)
    field_provenance = proposition.field_provenance + (optional_provenance,)
    forged = replace(
        proposition,
        proposition_id=deterministic_proposition_id(
            proposition.proposition_type,
            proposition.semantics_version,
            roles,
            proposition.temporal_scope,
            proposition.parameters_json,
            semantics_fingerprint=proposition.semantics_fingerprint,
            identity_algorithm_version=proposition.identity_algorithm_version,
        ),
        roles=roles,
        field_provenance=field_provenance,
    )
    validator = PropositionProvenanceValidator(
        repository, semantics_registry, construction_registry
    )

    with pytest.raises(
        PropositionProvenanceError,
        match="exact registered construction bindings",
    ):
        validator.validate(forged)


def _make_optional_role_constructor(
    repository,
    *,
    entity_required=False,
    parameter=None,
    parameter_binding_kind="event_attribute",
):
    base_semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    optional_event = PropositionRoleDefinition(
        role_name="device_event",
        role_kind="event",
        cardinality="one",
        event_type="device_activity",
        meaning="An optional device activity event.",
        required=False,
    )
    device_role = PropositionRoleDefinition(
        role_name="device",
        role_kind="entity",
        cardinality="one",
        entity_type="host",
        meaning="The device participating in the optional device activity.",
        required=entity_required,
    )
    semantics = replace(
        base_semantics,
        roles=base_semantics.roles + (optional_event, device_role),
        parameters=() if parameter is None else (parameter,),
        temporal_scope=None,
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(semantics)
    base_construction = build_default_proposition_construction_registry().get(
        "account_authentication_sequence", 1
    )
    construction = replace(
        base_construction,
        event_bindings=base_construction.event_bindings + (
            EventRoleBinding("device_event", "device_activity"),
        ),
        entity_bindings=base_construction.entity_bindings + (
            EntityRoleBinding(
                "device", (ParticipantRoleBinding("device_event", "device"),)
            ),
        ),
        event_attribute_bindings=(
            (
                EventAttributeParameterBinding(
                    parameter.parameter_name,
                    "device_event",
                    "state",
                    "Read the optional device state attribute.",
                ),
            )
            if parameter is not None and parameter_binding_kind == "event_attribute"
            else ()
        ),
        temporal_parameter_bindings=(
            (
                TemporalDifferenceParameterBinding(
                    parameter.parameter_name,
                    "success_event",
                    "device_event",
                    "Derive elapsed seconds to the optional device event.",
                ),
            )
            if parameter is not None and parameter_binding_kind == "temporal"
            else ()
        ),
        relationship_parameter_bindings=(
            (
                RelationshipParameterBinding(
                    parameter.parameter_name,
                    "account",
                    "device",
                    "account_uses_device",
                    "relationship_id",
                    "Use the explicit account-to-device relationship.",
                ),
            )
            if parameter is not None and parameter_binding_kind == "relationship"
            else ()
        ),
    )
    if parameter is not None and parameter_binding_kind not in {
        "event_attribute", "temporal", "relationship"
    }:
        raise ValueError("unsupported optional parameter binding kind")
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(construction)
    return PropositionConstructor(repository, semantics_registry, construction_registry)


def test_optional_event_and_dependent_entity_are_omitted_when_no_source_matches(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-absent", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    proposition = _make_optional_role_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )
    role_names = {item.role_name for item in proposition.roles}
    assert "device_event" not in role_names
    assert "device" not in role_names


def test_optional_event_and_dependent_entity_are_populated_when_source_matches(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-present", created_at=start,
    )
    host = Entity(
        entity_id=new_id(), entity_type="host",
        canonical_identifier="optional-host", created_at=start,
    )
    events, evidence_set = _make_authentication_evidence(repository, source, account, start)
    device_event = persist_event(
        repository, source, host, "device_activity", "device",
        start + timedelta(seconds=240),
    )
    evidence_set = replace(
        evidence_set,
        evidence=evidence_set.evidence + (EvidenceReference(event_id=device_event.event_id),),
    )
    proposition = _make_optional_role_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )
    roles = {item.role_name: item.value_id for item in proposition.roles}
    assert roles["device_event"] == device_event.event_id
    assert roles["device"] == host.entity_id


def test_multiple_matches_for_optional_event_are_rejected_as_ambiguous(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-ambiguous", created_at=start,
    )
    host = Entity(
        entity_id=new_id(), entity_type="host",
        canonical_identifier="optional-host-ambiguous", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    first = persist_event(
        repository, source, host, "device_activity", "device",
        start + timedelta(seconds=240),
    )
    second = persist_event(
        repository, source, host, "device_activity", "device",
        start + timedelta(seconds=250),
    )
    evidence_set = replace(
        evidence_set,
        evidence=evidence_set.evidence + (
            EvidenceReference(event_id=first.event_id),
            EvidenceReference(event_id=second.event_id),
        ),
    )
    with pytest.raises(
        PropositionConstructionError,
        match="expected at most one device_activity event for device_event; found 2",
    ):
        _make_optional_role_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_required_entity_depending_on_absent_optional_event_fails(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="required-dependent", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    with pytest.raises(
        PropositionConstructionError,
        match="required entity role device depends on an absent optional event role",
    ):
        _make_optional_role_constructor(
            repository, entity_required=True
        ).construct(evidence_set, "account_authentication_sequence", 1)


def test_optional_parameter_uses_explicit_default_when_optional_event_is_absent(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-default", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    parameter = SemanticParameterDefinition(
        "device_state", "string", required=False, default_value="unknown",
        meaning="State reported by the optional device event.",
    )
    proposition = _make_optional_role_constructor(
        repository, parameter=parameter
    ).construct(evidence_set, "account_authentication_sequence", 1)
    assert proposition.parameters_json == '{"device_state":"unknown"}'
    assert "device_state" not in {
        item.field_name for item in proposition.field_provenance
    }



def test_optional_temporal_parameter_is_omitted_when_event_dependency_is_absent(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-temporal-parameter", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    parameter = SemanticParameterDefinition(
        "success_to_device_seconds",
        "integer",
        required=False,
        meaning="Elapsed seconds from authentication success to optional device activity.",
    )
    proposition = _make_optional_role_constructor(
        repository,
        parameter=parameter,
        parameter_binding_kind="temporal",
    ).construct(evidence_set, "account_authentication_sequence", 1)

    assert proposition.parameters_json == "{}"
    assert "success_to_device_seconds" not in {
        item.field_name for item in proposition.field_provenance
    }


def test_optional_relationship_parameter_is_omitted_when_entity_dependency_is_absent(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-relationship-parameter", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    parameter = SemanticParameterDefinition(
        "account_device_relationship_id",
        "string",
        required=False,
        meaning="The explicit relationship connecting the account to the optional device.",
    )
    proposition = _make_optional_role_constructor(
        repository,
        parameter=parameter,
        parameter_binding_kind="relationship",
    ).construct(evidence_set, "account_authentication_sequence", 1)

    assert proposition.parameters_json == "{}"
    assert "account_device_relationship_id" not in {
        item.field_name for item in proposition.field_provenance
    }


def test_optional_parameter_without_default_is_omitted_when_event_source_is_absent(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="optional-no-default", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(repository, source, account, start)
    parameter = SemanticParameterDefinition(
        "device_state",
        "string",
        required=False,
        meaning="State reported by the optional device activity event.",
    )
    proposition = _make_optional_role_constructor(
        repository,
        parameter=parameter,
    ).construct(evidence_set, "account_authentication_sequence", 1)

    assert proposition.parameters_json == "{}"
    assert "device_state" not in {
        item.field_name for item in proposition.field_provenance
    }


@pytest.mark.parametrize("optional_offset_seconds", [-60, 240])
def test_optional_event_does_not_expand_implicit_temporal_scope(
    tmp_path, optional_offset_seconds
):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier=f"optional-scope-{optional_offset_seconds}",
        created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    optional_host = Entity(
        entity_id=new_id(), entity_type="host",
        canonical_identifier=f"optional-scope-host-{optional_offset_seconds}",
        created_at=start,
    )
    optional_event = persist_event(
        repository, source, optional_host, "device_activity", "device",
        start + timedelta(seconds=optional_offset_seconds),
    )
    evidence_set = replace(
        evidence_set,
        evidence=evidence_set.evidence + (
            EvidenceReference(event_id=optional_event.event_id),
        ),
    )

    proposition = _make_optional_role_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )

    assert proposition.temporal_scope.starts_at == start
    assert proposition.temporal_scope.ends_at == start + timedelta(seconds=180)


def test_validator_rejects_forged_fallback_scope_even_with_recalculated_id(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="forged-fallback-scope", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    base_semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(replace(base_semantics, temporal_scope=None))
    construction_registry = build_default_proposition_construction_registry()
    constructor = PropositionConstructor(
        repository, semantics_registry, construction_registry
    )
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    forged_scope = replace(
        proposition.temporal_scope,
        starts_at=proposition.temporal_scope.starts_at - timedelta(seconds=30),
    )
    forged_id = deterministic_proposition_id(
        proposition.proposition_type,
        proposition.semantics_version,
        proposition.roles,
        forged_scope,
        proposition.parameters_json,
        semantics_fingerprint=proposition.semantics_fingerprint,
        identity_algorithm_version=proposition.identity_algorithm_version,
    )
    forged = replace(
        proposition,
        proposition_id=forged_id,
        temporal_scope=forged_scope,
    )
    validator = PropositionProvenanceValidator(
        repository,
        constructor.semantics_registry,
        constructor.construction_registry,
    )

    with pytest.raises(
        PropositionProvenanceError,
        match="proposition temporal scope does not match semantic boundaries",
    ):
        validator.validate(forged)


def test_semantics_without_required_event_role_requires_explicit_scope():
    base = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    optional_event_roles = tuple(
        replace(role, required=False) if role.role_kind == "event" else role
        for role in base.roles
    )

    with pytest.raises(
        ValueError,
        match="semantics without an explicit temporal scope must declare at least one required event role",
    ):
        replace(
            base,
            roles=optional_event_roles,
            temporal_constraints=(),
            temporal_scope=None,
        )



def test_proposition_preserves_exact_immutable_evidence_selection_manifest(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="manifest-preservation", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )

    proposition = make_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )

    assert proposition.evidence_manifest.evidence_set_id == evidence_set.evidence_set_id
    assert proposition.evidence_manifest.evidence == evidence_set.evidence
    with pytest.raises(AttributeError):
        proposition.evidence_manifest.evidence_set_id = new_id()


def test_overlapping_observation_and_event_references_are_preserved_without_duplicate_events(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="overlapping-manifest", created_at=start,
    )
    events, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    overlapping_references = (
        EvidenceReference(observation_id=events[0].observation_id),
        EvidenceReference(event_id=events[0].event_id),
        *(
            EvidenceReference(event_id=item.event_id)
            for item in events[1:]
        ),
    )
    evidence_set = replace(
        evidence_set,
        evidence=overlapping_references,
    )

    proposition = make_constructor(repository).construct(
        evidence_set, "account_authentication_sequence", 1
    )

    assert proposition.evidence_manifest.evidence == overlapping_references
    assert proposition.evidence_manifest.evidence[0].observation_id == events[0].observation_id
    assert proposition.evidence_manifest.evidence[1].event_id == events[0].event_id
    assert tuple(
        item.value_id for item in proposition.roles
        if item.role_kind == "event"
    ) == tuple(item.event_id for item in events)


def test_validator_rejects_field_provenance_outside_evidence_manifest(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="manifest-boundary", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    constructor = make_constructor(repository)
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    omitted_event_id = proposition.field_provenance[0].event_ids[0]
    reduced_references = tuple(
        item for item in proposition.evidence_manifest.evidence
        if item.event_id != omitted_event_id
    )
    forged = replace(
        proposition,
        evidence_manifest=replace(
            proposition.evidence_manifest,
            evidence=reduced_references,
        ),
    )

    with pytest.raises(
        PropositionProvenanceError,
        match="expected exactly one|field provenance event is outside the evidence manifest",
    ):
        PropositionProvenanceValidator(
            repository,
            constructor.semantics_registry,
            constructor.construction_registry,
        ).validate(forged)


def test_validator_rejects_new_ambiguous_required_event_in_manifest(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="selection-required", created_at=start,
    )
    events, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    constructor = make_constructor(repository)
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    second_account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="selection-second", created_at=start,
    )
    duplicate = persist_event(
        repository, source, second_account, "authentication_failure", "target",
        start + timedelta(seconds=5),
    )
    forged = replace(
        proposition,
        evidence_manifest=replace(
            proposition.evidence_manifest,
            evidence=proposition.evidence_manifest.evidence
            + (EvidenceReference(event_id=duplicate.event_id),),
        ),
    )

    with pytest.raises(PropositionProvenanceError, match="expected at most one authentication_failure"):
        PropositionProvenanceValidator(
            repository, constructor.semantics_registry, constructor.construction_registry
        ).validate(forged)


def test_validator_rejects_omitted_optional_event_role_when_manifest_has_match(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="selection-optional", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    constructor = _make_optional_role_constructor(repository)
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    device = Entity(
        entity_id=new_id(), entity_type="host",
        canonical_identifier="selection-optional-device", created_at=start,
    )
    device_event = persist_event(
        repository, source, device, "device_activity", "device",
        start + timedelta(seconds=240),
    )
    forged = replace(
        proposition,
        evidence_manifest=replace(
            proposition.evidence_manifest,
            evidence=proposition.evidence_manifest.evidence
            + (EvidenceReference(event_id=device_event.event_id),),
        ),
    )

    with pytest.raises(PropositionProvenanceError, match="event role presence or value does not match manifest selection"):
        PropositionProvenanceValidator(
            repository, constructor.semantics_registry, constructor.construction_registry
        ).validate(forged)


def test_constructor_rejects_one_event_bound_to_multiple_semantic_roles(tmp_path):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(), entity_type="user_account",
        canonical_identifier="duplicate-role-binding", created_at=start,
    )
    _, evidence_set = _make_authentication_evidence(
        repository, source, account, start
    )
    base_semantics = build_default_proposition_semantics_registry().get(
        "account_authentication_sequence", 1
    )
    optional_duplicate = PropositionRoleDefinition(
        role_name="second_failure_event",
        role_kind="event",
        cardinality="one",
        event_type="authentication_failure",
        meaning="An additional optional failure-event role.",
        required=False,
    )
    semantics_registry = PropositionSemanticsRegistry()
    semantics_registry.register(
        replace(base_semantics, roles=base_semantics.roles + (optional_duplicate,))
    )
    base_construction = build_default_proposition_construction_registry().get(
        "account_authentication_sequence", 1
    )
    construction_registry = PropositionConstructionRegistry()
    construction_registry.register(
        replace(
            base_construction,
            event_bindings=base_construction.event_bindings
            + (EventRoleBinding("second_failure_event", "authentication_failure"),),
        )
    )
    constructor = PropositionConstructor(
        repository, semantics_registry, construction_registry
    )

    with pytest.raises(
        PropositionConstructionError,
        match="one source event cannot satisfy multiple semantic roles",
    ):
        constructor.construct(
            evidence_set, "account_authentication_sequence", 1
        )


def test_constructor_freezes_semantics_and_construction_registries(tmp_path):
    repository, _, _ = make_repository(tmp_path)
    constructor = make_constructor(repository)

    with pytest.raises(RuntimeError, match="registry is frozen"):
        constructor.semantics_registry.register(
            build_default_proposition_semantics_registry().get(
                "account_authentication_sequence", 1
            )
        )
    with pytest.raises(RuntimeError, match="registry is frozen"):
        constructor.construction_registry.register(
            build_default_proposition_construction_registry().get(
                "account_authentication_sequence", 1
            )
        )

def _repository_failure(failure):
    if failure == "key_error":
        return KeyError("simulated missing repository field")
    if failure == "sqlite_error":
        return sqlite3.OperationalError("simulated repository failure")
    if failure == "unicode_error":
        return UnicodeEncodeError(
            "utf-8", "\ud800", 0, 1, "simulated invalid Unicode"
        )
    if failure == "attribute_error":
        return AttributeError("simulated malformed repository record")
    if failure == "type_error":
        return TypeError("simulated malformed repository result")
    if failure == "missing_record":
        return None
    raise AssertionError(f"unknown failure case: {failure}")


@pytest.mark.parametrize(
    "failure",
    ["key_error", "sqlite_error", "unicode_error", "attribute_error", "type_error", "missing_record"],
)
def test_constructor_normalizes_expected_repository_boundary_failures(
    tmp_path, monkeypatch, failure
):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier=f"constructor-boundary-{failure}",
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
        persist_event(
            repository, source, account, "host_activity", "subject",
            start + timedelta(seconds=180),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=4),
    )
    original_error = _repository_failure(failure)

    def fail_get_event(_event_id):
        if original_error is None:
            return None
        raise original_error

    monkeypatch.setattr(repository, "get_event", fail_get_event)

    with pytest.raises(PropositionConstructionError) as captured:
        make_constructor(repository).construct(
            evidence_set, "account_authentication_sequence", 1
        )

    if original_error is None:
        assert captured.value.__cause__ is not None
    else:
        assert captured.value.__cause__ is original_error


@pytest.mark.parametrize(
    "failure",
    ["key_error", "sqlite_error", "unicode_error", "attribute_error", "type_error", "missing_record"],
)
def test_validator_normalizes_expected_repository_boundary_failures(
    tmp_path, monkeypatch, failure
):
    repository, source, start = make_repository(tmp_path)
    account = Entity(
        entity_id=new_id(),
        entity_type="user_account",
        canonical_identifier=f"validator-boundary-{failure}",
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
        persist_event(
            repository, source, account, "host_activity", "subject",
            start + timedelta(seconds=180),
        ),
    )
    evidence_set = EvidenceSet(
        evidence_set_id=new_id(),
        evidence=tuple(EvidenceReference(event_id=item.event_id) for item in events),
        created_at=start + timedelta(minutes=4),
    )
    constructor = make_constructor(repository)
    proposition = constructor.construct(
        evidence_set, "account_authentication_sequence", 1
    )
    original_error = _repository_failure(failure)

    def fail_get_event(_event_id):
        if original_error is None:
            return None
        raise original_error

    monkeypatch.setattr(repository, "get_event", fail_get_event)
    validator = PropositionProvenanceValidator(
        repository,
        constructor.semantics_registry,
        constructor.construction_registry,
    )

    with pytest.raises(PropositionProvenanceError) as captured:
        validator.validate(proposition)

    if original_error is None:
        assert captured.value.__cause__ is not None
    else:
        assert captured.value.__cause__ is original_error

