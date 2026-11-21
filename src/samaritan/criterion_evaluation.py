"""Deterministic execution of a finite registry of typed criterion evaluators."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import Event, Observation, normalize_timestamp
from samaritan.database import DatabaseIntegrityError
from samaritan.evaluation_contracts import EvidenceSet
from samaritan.reasoning_contracts import (
    CriterionDefinition,
    CriterionEvaluatorRegistry,
    CriterionFinding,
    CriterionOutcomeRegistry,
)
from samaritan.proposition_contracts import Proposition
from samaritan.repository import Repository


class CriterionEvaluationError(ValueError):
    """Raised when a criterion cannot be safely or unambiguously evaluated."""


class CriterionEvaluationEngine:
    """Execute only evaluator types explicitly supported by the built-in dispatcher."""

    def __init__(
        self,
        repository: Repository,
        evaluator_registry: CriterionEvaluatorRegistry,
        outcome_registry: CriterionOutcomeRegistry,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("repository must be a Repository")
        if not isinstance(evaluator_registry, CriterionEvaluatorRegistry):
            raise TypeError("evaluator_registry must be a CriterionEvaluatorRegistry")
        if not isinstance(outcome_registry, CriterionOutcomeRegistry):
            raise TypeError("outcome_registry must be a CriterionOutcomeRegistry")
        self.repository = repository
        self.evaluator_registry = evaluator_registry
        self.outcome_registry = outcome_registry
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def evaluate(
        self,
        criterion: CriterionDefinition,
        proposition: Proposition,
        evidence_set: EvidenceSet,
    ) -> CriterionFinding:
        """Produce a stance-neutral finding from registered evidence and evaluator types."""
        if not isinstance(criterion, CriterionDefinition):
            raise TypeError("criterion must be a CriterionDefinition")
        if not isinstance(proposition, Proposition):
            raise TypeError("proposition must be a Proposition")
        if not isinstance(evidence_set, EvidenceSet):
            raise TypeError("evidence_set must be an EvidenceSet")

        evaluator = self.evaluator_registry.validate_criterion(criterion)
        parameters = json.loads(criterion.parameters_json)
        events, _ = self._resolve_evidence(evidence_set.evidence)
        event_refs = tuple(
            reference
            for event in events.values()
            for reference in (
                EvidenceReference(event_id=event.event_id),
                EvidenceReference(observation_id=event.observation_id),
            )
        )
        if not events:
            return self._insufficient(
                criterion,
                "No events were available in the selected EvidenceSet.",
                ("At least one relevant event",),
            )

        if evaluator.evaluator_type == "event_attribute_equals":
            finding = self._event_attribute_equals(
                criterion, proposition, parameters, events, event_refs
            )
        elif evaluator.evaluator_type == "event_attribute_present":
            finding = self._event_attribute_present(
                criterion, proposition, parameters, events, event_refs
            )
        elif evaluator.evaluator_type == "event_type_count":
            finding = self._event_type_count(
                criterion, proposition, parameters, events, event_refs
            )
        elif evaluator.evaluator_type == "participant_role_matches_proposition_role":
            finding = self._participant_role_matches(
                criterion, proposition, parameters, events, event_refs
            )
        elif evaluator.evaluator_type == "temporal_gap_within":
            finding = self._temporal_gap_within(
                criterion, proposition, parameters, event_refs
            )
        else:
            raise CriterionEvaluationError(
                f"no deterministic implementation for evaluator type: {evaluator.evaluator_type}"
            )
        if finding.outcome_code is not None:
            self.outcome_registry.get(
                criterion.criterion_type,
                criterion.outcome_vocabulary_version,
                finding.outcome_code,
            )
        return finding

    def _event_for_role(
        self,
        proposition: Proposition,
        role_name: str,
        events: dict[str, Event],
    ) -> Event | None:
        role = next((item for item in proposition.roles if item.role_name == role_name), None)
        if role is None or role.role_kind != "event":
            return None
        return events.get(role.value_id)

    def _event_attribute_equals(
        self,
        criterion: CriterionDefinition,
        proposition: Proposition,
        parameters: dict,
        events: dict[str, Event],
        evidence: tuple[EvidenceReference, ...],
    ) -> CriterionFinding:
        event = self._event_for_role(proposition, parameters["event_role"], events)
        if event is None:
            return self._insufficient(
                criterion,
                "The proposition event role is not represented in the EvidenceSet.",
                (f"Evidence for event role {parameters['event_role']}",),
                evidence,
            )
        data = json.loads(event.normalized_data_json)
        if parameters["attribute_name"] not in data:
            return self._insufficient(
                criterion,
                f"Registered event attribute {parameters['attribute_name']} is absent from the normalized event.",
                (f"Normalized value for attribute {parameters['attribute_name']}",),
                (EvidenceReference(event_id=event.event_id), EvidenceReference(observation_id=event.observation_id)),
            )
        matched = (
            type(data[parameters["attribute_name"]]) is type(parameters["expected_value"])
            and data[parameters["attribute_name"]] == parameters["expected_value"]
        )
        return self._result(
            criterion,
            "matched" if matched else "not_matched",
            "equal" if matched else "not_equal",
            (EvidenceReference(event_id=event.event_id), EvidenceReference(observation_id=event.observation_id)),
            (
                f"Registered event attribute {parameters['attribute_name']} equals the expected value."
                if matched
                else f"Registered event attribute {parameters['attribute_name']} does not equal the expected value."
            ),
        )

    def _event_attribute_present(
        self,
        criterion: CriterionDefinition,
        proposition: Proposition,
        parameters: dict,
        events: dict[str, Event],
        evidence: tuple[EvidenceReference, ...],
    ) -> CriterionFinding:
        event = self._event_for_role(proposition, parameters["event_role"], events)
        if event is None:
            return self._insufficient(
                criterion,
                "The proposition event role is not represented in the EvidenceSet.",
                (f"Evidence for event role {parameters['event_role']}",),
                evidence,
            )
        present = parameters["attribute_name"] in json.loads(event.normalized_data_json)
        if not present:
            return self._insufficient(
                criterion,
                f"Registered event attribute {parameters['attribute_name']} is not represented in the normalized event.",
                (f"Validated coverage or normalized value for attribute {parameters['attribute_name']}",),
                (EvidenceReference(event_id=event.event_id), EvidenceReference(observation_id=event.observation_id)),
            )
        return self._result(
            criterion,
            "matched",
            "present",
            (EvidenceReference(event_id=event.event_id), EvidenceReference(observation_id=event.observation_id)),
            f"Registered event attribute {parameters['attribute_name']} is present.",
        )

    def _event_type_count(
        self,
        criterion: CriterionDefinition,
        proposition: Proposition,
        parameters: dict,
        events: dict[str, Event],
        evidence: tuple[EvidenceReference, ...],
    ) -> CriterionFinding:
        role_values = {item.role_name: item for item in proposition.roles}
        entity_role = role_values.get(parameters["entity_role"])
        if entity_role is None or entity_role.role_kind != "entity":
            return self._insufficient(
                criterion,
                "The entity role required to scope the event count is unavailable.",
                (f"A valid proposition entity role {parameters['entity_role']}",),
                evidence,
            )
        proposition_event_ids = {
            item.value_id for item in proposition.roles if item.role_kind == "event"
        }
        if not proposition_event_ids.issubset(events):
            return self._insufficient(
                criterion,
                "The EvidenceSet does not contain every event that defines the proposition.",
                ("Evidence for every proposition event role",),
                evidence,
            )

        def within_scope(event: Event) -> bool:
            scope = proposition.temporal_scope
            after_start = (
                event.occurred_at > scope.starts_at
                or (scope.start_inclusive and event.occurred_at == scope.starts_at)
            )
            before_end = (
                event.occurred_at < scope.ends_at
                or (scope.end_inclusive and event.occurred_at == scope.ends_at)
            )
            return after_start and before_end

        count = sum(
            event.event_type == parameters["event_type"]
            and within_scope(event)
            and any(
                participant.entity_id == entity_role.value_id
                for participant in event.participants
            )
            for event in events.values()
        )
        matched = count >= parameters["minimum_count"]
        return self._result(
            criterion,
            "matched" if matched else "not_matched",
            "threshold_met" if matched else "threshold_not_met",
            evidence,
            f"Observed {count} events of type {parameters['event_type']}; minimum required is {parameters['minimum_count']}.",
        )

    def _participant_role_matches(
        self,
        criterion: CriterionDefinition,
        proposition: Proposition,
        parameters: dict,
        events: dict[str, Event],
        evidence: tuple[EvidenceReference, ...],
    ) -> CriterionFinding:
        event = self._event_for_role(proposition, parameters["event_role"], events)
        entity_role = next(
            (item for item in proposition.roles if item.role_name == parameters["entity_role"]),
            None,
        )
        if event is None or entity_role is None or entity_role.role_kind != "entity":
            return self._insufficient(
                criterion,
                "The event or entity role required by the evaluator is unavailable.",
                ("A valid event role and entity role in the proposition",),
                evidence,
            )
        matches = [
            item.entity_id for item in event.participants
            if item.role == parameters["participant_role"]
        ]
        if len(matches) > 1:
            return self._insufficient(
                criterion,
                "The participant role is ambiguous.",
                (f"Exactly one participant with role {parameters['participant_role']}",),
                (EvidenceReference(event_id=event.event_id),),
            )
        if not matches:
            return self._insufficient(
                criterion,
                "The event contains no participant record for the requested role; absence cannot establish a mismatch without participant-coverage guarantees.",
                (f"Validated participant coverage for role {parameters['participant_role']}",),
                (EvidenceReference(event_id=event.event_id), EvidenceReference(observation_id=event.observation_id)),
            )
        matched = matches[0] == entity_role.value_id
        return self._result(
            criterion,
            "matched" if matched else "not_matched",
            "matches" if matched else "does_not_match",
            (EvidenceReference(event_id=event.event_id), EvidenceReference(observation_id=event.observation_id)),
            (
                "The event participant resolves to the proposition entity role."
                if matched
                else "The event participant does not resolve to the proposition entity role."
            ),
        )

    def _temporal_gap_within(
        self,
        criterion: CriterionDefinition,
        proposition: Proposition,
        parameters: dict,
        evidence: tuple[EvidenceReference, ...],
    ) -> CriterionFinding:
        role_values = {item.role_name: item for item in proposition.roles}
        earlier = role_values.get(parameters["earlier_event_role"])
        later = role_values.get(parameters["later_event_role"])
        if (
            earlier is None or later is None
            or earlier.role_kind != "event" or later.role_kind != "event"
            or earlier.occurred_at is None or later.occurred_at is None
        ):
            return self._insufficient(
                criterion,
                "The proposition does not contain both required event roles.",
                ("Both declared event roles with timestamps",),
                evidence,
            )
        gap = (later.occurred_at - earlier.occurred_at).total_seconds()
        if gap <= 0:
            return self._result(
                criterion,
                "not_matched",
                "ordering_violation",
                tuple(
                    item for item in evidence
                    if item.event_id in {earlier.value_id, later.value_id}
                ),
                "The earlier event role must strictly precede the later event role.",
            )
        matched = gap <= parameters["maximum_seconds"]
        relevant = tuple(
            item for item in evidence
            if item.event_id in {earlier.value_id, later.value_id}
        )
        if len(relevant) < 2:
            return self._insufficient(
                criterion,
                "The selected EvidenceSet does not contain both proposition events.",
                ("Evidence for both temporal boundary events",),
                evidence,
            )
        return self._result(
            criterion,
            "matched" if matched else "not_matched",
            "within_limit" if matched else "exceeds_limit",
            relevant,
            f"Observed temporal gap is {gap:g} seconds; maximum permitted is {parameters['maximum_seconds']} seconds.",
        )

    def _result(
        self,
        criterion: CriterionDefinition,
        status: str,
        outcome_code: str,
        evidence: tuple[EvidenceReference, ...],
        rationale: str,
    ) -> CriterionFinding:
        canonical_evidence = tuple(sorted(
            set(evidence),
            key=lambda item: (item.observation_id or "", item.event_id or ""),
        ))
        finding_id = self._deterministic_finding_id(
            criterion, "insufficient", None, canonical_evidence
        )
        return CriterionFinding(
            finding_id=finding_id,
            criterion_id=criterion.criterion_id,
            status=status,
            outcome_code=outcome_code,
            evidence=canonical_evidence,
            rationale=rationale,
            created_at=normalize_timestamp(self.clock(), "clock timestamp"),
        )

    def _insufficient(
        self,
        criterion: CriterionDefinition,
        rationale: str,
        missing_information: tuple[str, ...],
        evidence: tuple[EvidenceReference, ...] = (),
    ) -> CriterionFinding:
        canonical_evidence = tuple(sorted(
            set(evidence),
            key=lambda item: (item.observation_id or "", item.event_id or ""),
        ))
        finding_id = self._deterministic_finding_id(
            criterion, status, outcome_code, canonical_evidence
        )
        return CriterionFinding(
            finding_id=finding_id,
            criterion_id=criterion.criterion_id,
            status="insufficient",
            outcome_code=None,
            evidence=canonical_evidence,
            rationale=rationale,
            missing_information=missing_information,
            created_at=normalize_timestamp(self.clock(), "clock timestamp"),
        )

    @staticmethod
    def _deterministic_finding_id(
        criterion: CriterionDefinition,
        status: str,
        outcome_code: str | None,
        evidence: tuple[EvidenceReference, ...],
    ) -> str:
        payload = json.dumps(
            {
                "criterion_id": criterion.criterion_id,
                "criterion_type": criterion.criterion_type,
                "evaluator_version": criterion.evaluator_version,
                "parameters": json.loads(criterion.parameters_json),
                "status": status,
                "outcome_code": outcome_code,
                "evidence": [
                    {"observation_id": item.observation_id, "event_id": item.event_id}
                    for item in evidence
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            "samaritan:criterion-finding:v1:" + payload,
        ))

    def _resolve_evidence(
        self,
        references: tuple[EvidenceReference, ...],
    ) -> tuple[dict[str, Event], dict[str, Observation]]:
        events: dict[str, Event] = {}
        observations: dict[str, Observation] = {}
        for reference in references:
            if reference.event_id is not None:
                event = self.repository.get_event(reference.event_id)
                if event is None:
                    raise DatabaseIntegrityError(f"referenced event does not exist: {reference.event_id}")
                if not isinstance(event, Event) or event.event_id != reference.event_id:
                    raise DatabaseIntegrityError("repository returned an invalid or mismatched event")
                events[event.event_id] = event
                self._load_observation(event.observation_id, observations)
            else:
                observation = self._load_observation(reference.observation_id, observations)
                offset = 0
                while True:
                    related = self.repository.list_events(
                        observation_id=observation.observation_id,
                        limit=500,
                        offset=offset,
                    )
                    if not isinstance(related, tuple) or not all(
                        isinstance(item, Event) for item in related
                    ):
                        raise DatabaseIntegrityError("repository returned invalid events for an observation")
                    for event in related:
                        if event.observation_id != observation.observation_id:
                            raise DatabaseIntegrityError("event observation provenance mismatch")
                        events[event.event_id] = event
                    if len(related) < 500:
                        break
                    offset += 500
        return events, observations

    def _load_observation(
        self,
        observation_id: str,
        cache: dict[str, Observation],
    ) -> Observation:
        observation = cache.get(observation_id)
        if observation is None:
            observation = self.repository.get_observation(observation_id)
            if observation is None:
                raise DatabaseIntegrityError(f"referenced observation does not exist: {observation_id}")
            if not isinstance(observation, Observation) or observation.observation_id != observation_id:
                raise DatabaseIntegrityError("repository returned an invalid or mismatched observation")
            cache[observation_id] = observation
        return observation
