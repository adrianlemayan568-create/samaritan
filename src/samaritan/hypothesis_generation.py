"""Evidence-backed hypothesis generation from correlated event patterns."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from samaritan.analysis_contracts import (
    EvidenceReference,
    Hypothesis,
)
from samaritan.contracts import (
    Entity,
    Event,
    Observation,
    new_id,
)
from samaritan.correlation import (
    derive_correlation_facts,
    resolve_participant_entities,
)
from samaritan.correlation_contracts import Correlation, CorrelationRule
from samaritan.database import DatabaseIntegrityError
from samaritan.hypothesis_generation_contracts import (
    HypothesisGeneration,
    HypothesisGenerationResult,
    HypothesisGenerationRule,
)
from samaritan.hypothesis_patterns import match_hypothesis_pattern
from samaritan.repository import Repository


class HypothesisGenerationEngine:
    """Generate candidate hypotheses from validated evidence patterns."""

    def __init__(
        self,
        repository: Repository,
        rules: Iterable[HypothesisGenerationRule],
        *,
        correlation_rules: Iterable[CorrelationRule],
    ) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("repository must be a Repository instance")

        if isinstance(rules, (str, bytes)):
            raise TypeError(
                "rules must be an iterable of HypothesisGenerationRule objects"
            )

        try:
            configured_rules = tuple(rules)
        except TypeError as exc:
            raise TypeError("rules must be iterable") from exc

        if not configured_rules:
            raise ValueError(
                "at least one hypothesis generation rule is required"
            )

        if not all(
            isinstance(rule, HypothesisGenerationRule)
            for rule in configured_rules
        ):
            raise TypeError(
                "rules must contain only HypothesisGenerationRule objects"
            )

        rule_ids = tuple(rule.rule_id for rule in configured_rules)

        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("duplicate hypothesis generation rule IDs")

        if isinstance(correlation_rules, (str, bytes)):
            raise TypeError(
                "correlation_rules must be an iterable of CorrelationRule objects"
            )

        try:
            configured_correlation_rules = tuple(correlation_rules)
        except TypeError as exc:
            raise TypeError("correlation_rules must be iterable") from exc

        if not configured_correlation_rules:
            raise ValueError("at least one correlation rule is required")

        if not all(
            isinstance(rule, CorrelationRule)
            for rule in configured_correlation_rules
        ):
            raise TypeError(
                "correlation_rules must contain only CorrelationRule objects"
            )

        correlation_rule_ids = tuple(
            rule.rule_id for rule in configured_correlation_rules
        )

        if len(correlation_rule_ids) != len(set(correlation_rule_ids)):
            raise ValueError("duplicate correlation rule IDs")

        correlation_rule_map = {
            rule.rule_id: rule for rule in configured_correlation_rules
        }
        for generation_rule in configured_rules:
            role_map = {
                role.role_name: role
                for role in generation_rule.pattern.event_roles
            }
            bindings_by_role = {
                role.role_name: tuple(
                    binding
                    for binding in generation_rule.pattern.entity_bindings
                    if binding.event_role == role.role_name
                )
                for role in generation_rule.pattern.event_roles
            }
            for constraint in generation_rule.pattern.correlation_constraints:
                correlation_rule = correlation_rule_map.get(
                    constraint.correlation_rule_id
                )
                if correlation_rule is None:
                    raise ValueError(
                        "pattern references an unconfigured correlation rule: "
                        f"{constraint.correlation_rule_id}"
                    )

                first_role = role_map[constraint.first_event_role]
                second_role = role_map[constraint.second_event_role]
                expected_pair = frozenset(
                    (first_role.event_type, second_role.event_type)
                )
                configured_pairs = {
                    frozenset(pair)
                    for pair in correlation_rule.event_type_pairs
                }
                if expected_pair not in configured_pairs:
                    raise ValueError(
                        "pattern event types do not match their configured "
                        f"correlation rule: {correlation_rule.rule_id}"
                    )

                common_binding_names = (
                    {
                        binding.binding_name
                        for binding in bindings_by_role[first_role.role_name]
                    }
                    & {
                        binding.binding_name
                        for binding in bindings_by_role[second_role.role_name]
                    }
                )
                common_entity_types = {
                    binding.entity_type
                    for binding in generation_rule.pattern.entity_bindings
                    if binding.binding_name in common_binding_names
                }
                if not common_entity_types.intersection(
                    correlation_rule.shared_entity_types
                ):
                    raise ValueError(
                        "pattern identity bindings do not match the shared "
                        "entity types of correlation rule: "
                        f"{correlation_rule.rule_id}"
                    )

        self.repository = repository
        self.rules = tuple(sorted(configured_rules, key=lambda item: item.rule_id))
        self.correlation_rules = configured_correlation_rules

    def generate(
        self,
        correlations: Iterable[Correlation],
    ) -> HypothesisGenerationResult:
        """Generate hypotheses from validated correlation evidence.

        Every supplied correlation is checked for evidence integrity,
        including correlations that do not match a configured generation
        rule.

        Evidence records are retrieved at most once per ID during this
        invocation. No evidence or hypothesis is persisted or modified.
        """
        if isinstance(correlations, (str, bytes)):
            raise TypeError(
                "correlations must be an iterable of Correlation objects"
            )

        try:
            supplied = tuple(correlations)
        except TypeError as exc:
            raise TypeError("correlations must be iterable") from exc

        if not all(
            isinstance(item, Correlation)
            for item in supplied
        ):
            raise TypeError(
                "correlations must contain only Correlation objects"
            )

        correlation_ids = tuple(
            item.correlation_id for item in supplied
        )

        if len(correlation_ids) != len(set(correlation_ids)):
            raise ValueError("duplicate correlation IDs are not allowed")

        event_pairs = tuple(
            frozenset(item.event_ids) for item in supplied
        )

        if len(event_pairs) != len(set(event_pairs)):
            raise ValueError(
                "duplicate unordered event pairs are not allowed"
            )

        event_cache: dict[str, Event] = {}
        observation_cache: dict[str, Observation] = {}

        entities = self._validate_correlation_evidence(
            supplied,
            event_cache,
            observation_cache,
        )
        persisted_events = tuple(
            event_cache[event_id] for event_id in sorted(event_cache)
        )

        hypotheses = []
        generations = []

        for rule in self.rules:
            matches = match_hypothesis_pattern(
                rule.pattern,
                supplied,
                persisted_events,
                entities,
            )

            for match in matches:
                matched_events = tuple(
                    event for _, event in match.events
                )
                event_ids = tuple(
                    sorted(event.event_id for event in matched_events)
                )
                observation_ids = tuple(
                    sorted({
                        event.observation_id for event in matched_events
                    })
                )

                evidence = tuple(
                    [
                        EvidenceReference(event_id=event_id)
                        for event_id in event_ids
                    ]
                    + [
                        EvidenceReference(observation_id=observation_id)
                        for observation_id in observation_ids
                    ]
                )

                created_at = datetime.now(timezone.utc)
                hypothesis_id = new_id()

                hypothesis = Hypothesis(
                    hypothesis_id=hypothesis_id,
                    statement=rule.statement,
                    description=rule.description,
                    created_at=created_at,
                )

                generation = HypothesisGeneration(
                    generation_id=new_id(),
                    hypothesis_id=hypothesis_id,
                    rule_id=rule.rule_id,
                    correlation_ids=tuple(
                        sorted(
                            correlation.correlation_id
                            for correlation in match.correlations
                        )
                    ),
                    evidence=evidence,
                    created_at=created_at,
                )

                hypotheses.append(hypothesis)
                generations.append(generation)

        return HypothesisGenerationResult(
            hypotheses=tuple(hypotheses),
            generations=tuple(generations),
        )

    def _validate_correlation_evidence(
        self,
        correlations: tuple[Correlation, ...],
        event_cache: dict[str, Event],
        observation_cache: dict[str, Observation],
    ) -> dict[str, Entity]:
        """Verify every correlation's event and observation provenance."""
        for correlation in correlations:
            for event_id, observation_id in zip(
                correlation.event_ids,
                correlation.observation_ids,
            ):
                event = event_cache.get(event_id)

                if event is None:
                    event = self._get_required_event(event_id)
                    event_cache[event_id] = event

                if event.event_id != event_id:
                    raise DatabaseIntegrityError(
                        "repository returned an event with an "
                        f"unexpected ID: {event_id}"
                    )

                if event.observation_id != observation_id:
                    raise DatabaseIntegrityError(
                        "correlation observation reference does not "
                        f"match event provenance: {event_id}"
                    )

                observation = observation_cache.get(observation_id)

                if observation is None:
                    observation = self._get_required_observation(
                        observation_id
                    )
                    observation_cache[observation_id] = observation

                if observation.observation_id != observation_id:
                    raise DatabaseIntegrityError(
                        "repository returned an observation with an "
                        f"unexpected ID: {observation_id}"
                    )

        entities = resolve_participant_entities(
            self.repository,
            tuple(event_cache.values()),
        )

        for correlation in correlations:
            first_event = event_cache[correlation.event_ids[0]]
            second_event = event_cache[correlation.event_ids[1]]
            facts = derive_correlation_facts(
                first_event,
                second_event,
                entities,
                self.correlation_rules,
            )

            if facts is None:
                raise DatabaseIntegrityError(
                    "correlation claims no match under the configured "
                    "correlation rules"
                )

            expected_shared_ids, expected_rules, expected_time_delta = facts

            if correlation.shared_entity_ids != expected_shared_ids:
                raise DatabaseIntegrityError(
                    "correlation shared entity claims do not match "
                    "persisted event participants"
                )

            if correlation.time_delta_seconds != expected_time_delta:
                raise DatabaseIntegrityError(
                    "correlation time delta does not match persisted events"
                )

            if correlation.matched_rules != expected_rules:
                raise DatabaseIntegrityError(
                    "correlation matched rules do not match configured "
                    "correlation rules and persisted events"
                )

        return entities

    def _get_required_event(self, event_id: str) -> Event:
        """Retrieve an event or raise an integrity error if missing."""
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

    def _get_required_observation(
        self,
        observation_id: str,
    ) -> Observation:
        """Retrieve an observation or raise an integrity error if missing."""
        observation = self.repository.get_observation(observation_id)

        if observation is None:
            raise DatabaseIntegrityError(
                "referenced observation does not exist: "
                f"{observation_id}"
            )

        if not isinstance(observation, Observation):
            raise DatabaseIntegrityError(
                "repository returned an invalid observation: "
                f"{observation_id}"
            )

        return observation
