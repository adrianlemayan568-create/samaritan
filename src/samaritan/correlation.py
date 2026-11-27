"""Evidence-preserving pairwise event correlation."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import combinations

from samaritan.contracts import Entity, Event, new_id, validate_id
from samaritan.correlation_contracts import Correlation, CorrelationRule
from samaritan.database import DatabaseIntegrityError
from samaritan.repository import Repository


MAX_EVENTS_PER_BATCH = 100


def resolve_participant_entities(
    repository: Repository,
    events: tuple[Event, ...],
) -> dict[str, Entity]:
    """Resolve every participant entity for the supplied persisted events."""
    requested_ids = tuple(
        dict.fromkeys(
            participant.entity_id
            for event in events
            for participant in event.participants
        )
    )

    if not requested_ids:
        return {}

    retrieved_entities = repository.get_entities_by_ids(requested_ids)

    if not isinstance(retrieved_entities, tuple) or not all(
        isinstance(entity, Entity)
        for entity in retrieved_entities
    ):
        raise DatabaseIntegrityError(
            "repository returned invalid participant entity records"
        )

    entities = {
        entity.entity_id: entity
        for entity in retrieved_entities
    }

    missing_ids = tuple(
        entity_id
        for entity_id in requested_ids
        if entity_id not in entities
    )

    if missing_ids:
        raise DatabaseIntegrityError(
            "referenced participant entities do not exist: "
            + ", ".join(missing_ids)
        )

    return entities


def derive_correlation_facts(
    first_event: Event,
    second_event: Event,
    entities: dict[str, Entity],
    rules: tuple[CorrelationRule, ...],
) -> tuple[tuple[str, ...], tuple[CorrelationRule, ...], float] | None:
    """Derive shared entities, matching rules, and time delta from source records."""
    first_entity_ids = {
        participant.entity_id
        for participant in first_event.participants
    }
    second_entity_ids = {
        participant.entity_id
        for participant in second_event.participants
    }
    shared_entity_ids = first_entity_ids & second_entity_ids

    if not shared_entity_ids:
        return None

    time_delta = abs(
        (first_event.occurred_at - second_event.occurred_at).total_seconds()
    )
    event_type_pair = frozenset(
        (first_event.event_type, second_event.event_type)
    )

    matched_rules = []
    qualifying_entity_ids = set()

    for rule in rules:
        allowed_pairs = {
            frozenset(pair)
            for pair in rule.event_type_pairs
        }

        if event_type_pair not in allowed_pairs:
            continue

        if time_delta > rule.max_time_delta_seconds:
            continue

        allowed_entity_types = set(rule.shared_entity_types)
        rule_shared_ids = {
            entity_id
            for entity_id in shared_entity_ids
            if entities[entity_id].entity_type in allowed_entity_types
        }

        if not rule_shared_ids:
            continue

        matched_rules.append(rule)
        qualifying_entity_ids.update(rule_shared_ids)

    if not matched_rules:
        return None

    ordered_shared_ids = tuple(
        participant.entity_id
        for participant in first_event.participants
        if participant.entity_id in qualifying_entity_ids
    )
    ordered_shared_ids = tuple(dict.fromkeys(ordered_shared_ids))

    return ordered_shared_ids, tuple(matched_rules), time_delta


class CorrelationEngine:
    """Correlate events using explicit, configurable evidence rules."""

    def __init__(
        self,
        repository: Repository,
        rules: Iterable[CorrelationRule],
    ) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("repository must be a Repository instance")

        if isinstance(rules, (str, bytes)):
            raise TypeError("rules must be an iterable of CorrelationRule objects")

        try:
            configured_rules = tuple(rules)
        except TypeError as exc:
            raise TypeError("rules must be iterable") from exc

        if not configured_rules:
            raise ValueError("at least one correlation rule is required")

        if not all(
            isinstance(rule, CorrelationRule)
            for rule in configured_rules
        ):
            raise TypeError(
                "rules must contain only CorrelationRule objects"
            )

        rule_ids = tuple(rule.rule_id for rule in configured_rules)

        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("duplicate correlation rule IDs")

        self.repository = repository
        self.rules = configured_rules

    def correlate_pair(
        self,
        first_event_id: str,
        second_event_id: str,
    ) -> tuple[Correlation, ...]:
        """Correlate exactly two existing events."""
        first_event_id = validate_id(first_event_id, "first_event_id")
        second_event_id = validate_id(second_event_id, "second_event_id")

        if first_event_id == second_event_id:
            raise ValueError("a correlation requires two distinct events")

        first_event = self._get_required_event(first_event_id)
        second_event = self._get_required_event(second_event_id)

        events = (first_event, second_event)
        entities = self._resolve_participant_entities(events)

        correlation = self._correlate_loaded_pair(
            first_event,
            second_event,
            entities,
        )

        return (correlation,) if correlation is not None else ()

    def correlate_events(
        self,
        event_ids: Iterable[str],
    ) -> tuple[Correlation, ...]:
        """Correlate every unique event pair in a bounded input collection.

        Input order determines pair order. Duplicate event IDs are
        removed while preserving their first occurrence.
        """
        if isinstance(event_ids, (str, bytes)):
            raise TypeError("event_ids must be an iterable of IDs")

        try:
            supplied_ids = tuple(event_ids)
        except TypeError as exc:
            raise TypeError("event_ids must be iterable") from exc

        validated_ids = tuple(
            validate_id(item, "event_id")
            for item in supplied_ids
        )

        unique_ids = tuple(dict.fromkeys(validated_ids))

        if len(unique_ids) > MAX_EVENTS_PER_BATCH:
            raise ValueError(
                f"event_ids cannot contain more than "
                f"{MAX_EVENTS_PER_BATCH} unique events"
            )

        events = tuple(
            self._get_required_event(event_id)
            for event_id in unique_ids
        )

        if len(events) < 2:
            return ()

        entities = self._resolve_participant_entities(events)

        correlations = []

        for first_event, second_event in combinations(events, 2):
            correlation = self._correlate_loaded_pair(
                first_event,
                second_event,
                entities,
            )

            if correlation is not None:
                correlations.append(correlation)

        return tuple(correlations)

    def _get_required_event(self, event_id: str) -> Event:
        """Retrieve an event or raise an integrity error if it is missing."""
        event = self.repository.get_event(event_id)

        if event is None:
            raise DatabaseIntegrityError(
                f"referenced event does not exist: {event_id}"
            )

        if not isinstance(event, Event):
            raise DatabaseIntegrityError(
                f"repository returned an invalid event: {event_id}"
            )

        return event

    def _resolve_participant_entities(
        self,
        events: tuple[Event, ...],
    ) -> dict[str, Entity]:
        """Resolve all participant entities in one batch retrieval."""
        return resolve_participant_entities(self.repository, events)

    def _correlate_loaded_pair(
        self,
        first_event: Event,
        second_event: Event,
        entities: dict[str, Entity],
    ) -> Correlation | None:
        """Evaluate configured rules against an already loaded event pair."""
        facts = derive_correlation_facts(
            first_event,
            second_event,
            entities,
            self.rules,
        )

        if facts is None:
            return None

        ordered_shared_ids, matched_rules, time_delta = facts

        shared_entity_types = tuple(
            sorted(
                {
                    entities[entity_id].entity_type
                    for entity_id in ordered_shared_ids
                }
            )
        )

        matched_rule_ids = ", ".join(
            rule.rule_id for rule in matched_rules
        )

        explanation = (
            f"Events '{first_event.event_type}' and "
            f"'{second_event.event_type}' share "
            f"{len(ordered_shared_ids)} participant entity "
            f"(types: {', '.join(shared_entity_types)}) and occurred "
            f"{time_delta:g} seconds apart. "
            f"Matched rules: {matched_rule_ids}. "
            "This establishes shared-entity and temporal overlap only; "
            "it does not establish causation."
        )

        return Correlation(
            correlation_id=new_id(),
            event_ids=(
                first_event.event_id,
                second_event.event_id,
            ),
            observation_ids=(
                first_event.observation_id,
                second_event.observation_id,
            ),
            shared_entity_ids=ordered_shared_ids,
            matched_rules=tuple(matched_rules),
            time_delta_seconds=time_delta,
            explanation=explanation,
            created_at=datetime.now(timezone.utc),
        )
