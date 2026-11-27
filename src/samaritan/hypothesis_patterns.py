"""Deterministic semantic matching for declarative evidence patterns."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product

from samaritan.contracts import Entity, Event
from samaritan.correlation_contracts import Correlation
from samaritan.database import DatabaseIntegrityError
from samaritan.hypothesis_generation_contracts import (
    HypothesisPattern,
)


@dataclass(frozen=True, slots=True)
class HypothesisPatternMatch:
    """One complete assignment of persisted events to named pattern roles."""

    events: tuple[tuple[str, Event], ...]
    correlations: tuple[Correlation, ...]


def match_hypothesis_pattern(
    pattern: HypothesisPattern,
    correlations: Iterable[Correlation],
    events: Iterable[Event],
    entities: Mapping[str, Entity],
) -> tuple[HypothesisPatternMatch, ...]:
    """Return complete, unambiguous matches using authenticated source records.

    Correlation graph connectivity is not treated as proof of a pattern.
    Every role, participant binding, exact edge, identity continuity rule,
    and temporal constraint is checked independently.
    """
    if not isinstance(pattern, HypothesisPattern):
        raise TypeError("pattern must be a HypothesisPattern")
    if isinstance(correlations, (str, bytes)):
        raise TypeError("correlations must be an iterable of Correlation objects")
    if isinstance(events, (str, bytes)):
        raise TypeError("events must be an iterable of Event objects")
    if not isinstance(entities, Mapping):
        raise TypeError("entities must be a mapping of entity IDs to Entity objects")

    try:
        supplied_correlations = tuple(correlations)
        supplied_events = tuple(events)
    except TypeError as exc:
        raise TypeError("correlations and events must be iterable") from exc

    if not all(isinstance(item, Correlation) for item in supplied_correlations):
        raise TypeError("correlations must contain only Correlation objects")
    if not all(isinstance(item, Event) for item in supplied_events):
        raise TypeError("events must contain only Event objects")
    if not all(
        isinstance(key, str) and isinstance(value, Entity)
        and key == value.entity_id
        for key, value in entities.items()
    ):
        raise TypeError("entities must map entity IDs to matching Entity objects")

    correlation_ids = tuple(item.correlation_id for item in supplied_correlations)
    if len(correlation_ids) != len(set(correlation_ids)):
        raise ValueError("duplicate correlation IDs are not allowed")
    event_pairs = tuple(frozenset(item.event_ids) for item in supplied_correlations)
    if len(event_pairs) != len(set(event_pairs)):
        raise ValueError("duplicate unordered event pairs are not allowed")

    event_map = {event.event_id: event for event in supplied_events}
    if len(event_map) != len(supplied_events):
        raise ValueError("duplicate event IDs are not allowed")

    for correlation in supplied_correlations:
        for event_id in correlation.event_ids:
            if event_id not in event_map:
                raise DatabaseIntegrityError(
                    f"correlation references an event not supplied to pattern matching: {event_id}"
                )

    correlation_by_pair = {
        frozenset(item.event_ids): item for item in supplied_correlations
    }

    roles = pattern.event_roles
    bindings_by_role: dict[str, tuple] = {
        role.role_name: tuple(
            binding
            for binding in pattern.entity_bindings
            if binding.event_role == role.role_name
        )
        for role in roles
    }

    candidate_events: dict[str, tuple[Event, ...]] = {}
    candidate_bindings: dict[tuple[str, str], dict[str, str]] = {}

    for role in roles:
        candidates = []
        for event in sorted(supplied_events, key=lambda item: item.event_id):
            if event.event_type != role.event_type:
                continue

            resolved_bindings: dict[str, str] = {}
            valid = True
            for binding in bindings_by_role[role.role_name]:
                participants = tuple(
                    participant
                    for participant in event.participants
                    if participant.role == binding.participant_role
                )
                if len(participants) != 1:
                    valid = False
                    break

                entity_id = participants[0].entity_id
                entity = entities.get(entity_id)
                if entity is None:
                    raise DatabaseIntegrityError(
                        "persisted event references an unresolved participant entity: "
                        f"{entity_id}"
                    )
                if entity.entity_type != binding.entity_type:
                    valid = False
                    break
                resolved_bindings[binding.binding_name] = entity_id

            if valid:
                candidates.append(event)
                candidate_bindings[(role.role_name, event.event_id)] = (
                    resolved_bindings
                )
        candidate_events[role.role_name] = tuple(candidates)

    if any(not candidate_events[role.role_name] for role in roles):
        return ()

    role_names = tuple(role.role_name for role in roles)
    temporal_constraints = pattern.temporal_constraints
    possible: list[HypothesisPatternMatch] = []

    for assignment in product(
        *(candidate_events[role_name] for role_name in role_names)
    ):
        assigned_ids = tuple(event.event_id for event in assignment)
        if len(assigned_ids) != len(set(assigned_ids)):
            continue

        event_by_role = dict(zip(role_names, assignment))

        binding_values: dict[str, str] = {}
        binding_valid = True
        for role_name, event in zip(role_names, assignment):
            for binding_name, entity_id in candidate_bindings[
                (role_name, event.event_id)
            ].items():
                previous = binding_values.get(binding_name)
                if previous is not None and previous != entity_id:
                    binding_valid = False
                    break
                binding_values[binding_name] = entity_id
            if not binding_valid:
                break
        if not binding_valid:
            continue

        matched_correlations: dict[str, Correlation] = {}
        correlation_valid = True
        for constraint in pattern.correlation_constraints:
            first = event_by_role[constraint.first_event_role]
            second = event_by_role[constraint.second_event_role]
            correlation = correlation_by_pair.get(
                frozenset((first.event_id, second.event_id))
            )
            if correlation is None:
                correlation_valid = False
                break

            matched_rule = next(
                (
                    rule for rule in correlation.matched_rules
                    if rule.rule_id == constraint.correlation_rule_id
                ),
                None,
            )
            if matched_rule is None:
                correlation_valid = False
                break

            common_binding_names = (
                {
                    binding.binding_name
                    for binding in bindings_by_role[constraint.first_event_role]
                }
                & {
                    binding.binding_name
                    for binding in bindings_by_role[constraint.second_event_role]
                }
            )
            shared_binding_ids = {
                binding_values[name]
                for name in common_binding_names
                if binding_values[name] in correlation.shared_entity_ids
                and any(
                    binding.binding_name == name
                    and binding.entity_type in matched_rule.shared_entity_types
                    for binding in pattern.entity_bindings
                )
            }
            if not shared_binding_ids:
                correlation_valid = False
                break

            matched_correlations[correlation.correlation_id] = correlation

        if not correlation_valid:
            continue

        temporal_valid = True
        for constraint in temporal_constraints:
            earlier = event_by_role[constraint.earlier_event_role]
            later = event_by_role[constraint.later_event_role]
            delta = (later.occurred_at - earlier.occurred_at).total_seconds()

            if constraint.strict:
                if delta <= 0:
                    temporal_valid = False
                    break
            elif delta < 0:
                temporal_valid = False
                break

            maximum = constraint.maximum_interval_seconds
            if maximum is not None and delta > maximum:
                temporal_valid = False
                break

        if not temporal_valid:
            continue

        possible.append(
            HypothesisPatternMatch(
                events=tuple(zip(role_names, assignment)),
                correlations=tuple(
                    sorted(
                        matched_correlations.values(),
                        key=lambda item: item.correlation_id,
                    )
                ),
            )
        )

    possible.sort(
        key=lambda match: tuple(event.event_id for _, event in match.events)
    )

    # If two complete assignments agree in every role except one, the
    # differing role has multiple competing candidates. Reject both.
    ambiguous_indices: set[int] = set()
    for first_index, first_match in enumerate(possible):
        first_ids = tuple(event.event_id for _, event in first_match.events)
        for second_index in range(first_index + 1, len(possible)):
            second_ids = tuple(
                event.event_id for _, event in possible[second_index].events
            )
            differences = sum(
                first_id != second_id
                for first_id, second_id in zip(first_ids, second_ids)
            )
            same_event_set_with_role_permutation = (
                first_ids != second_ids
                and set(first_ids) == set(second_ids)
            )
            if differences == 1 or same_event_set_with_role_permutation:
                ambiguous_indices.add(first_index)
                ambiguous_indices.add(second_index)

    return tuple(
        match
        for index, match in enumerate(possible)
        if index not in ambiguous_indices
    )
