"""Evidence-backed hypothesis generation from correlated event patterns."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from samaritan.analysis_contracts import (
    EvidenceReference,
    Hypothesis,
)
from samaritan.contracts import (
    Event,
    Observation,
    new_id,
)
from samaritan.correlation_contracts import Correlation
from samaritan.database import DatabaseIntegrityError
from samaritan.hypothesis_generation_contracts import (
    HypothesisGeneration,
    HypothesisGenerationResult,
    HypothesisGenerationRule,
)
from samaritan.hypothesis_patterns import match_correlation_components
from samaritan.repository import Repository


class HypothesisGenerationEngine:
    """Generate candidate hypotheses from validated evidence patterns."""

    def __init__(
        self,
        repository: Repository,
        rules: Iterable[HypothesisGenerationRule],
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

        self.repository = repository
        self.rules = configured_rules

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

        self._validate_correlation_evidence(
            supplied,
            event_cache,
            observation_cache,
        )

        hypotheses = []
        generations = []

        for rule in self.rules:
            components = match_correlation_components(
                supplied,
                rule,
            )

            for component in components:
                event_ids = sorted({
                    event_id
                    for correlation in component
                    for event_id in correlation.event_ids
                })

                observation_ids = sorted({
                    observation_id
                    for correlation in component
                    for observation_id in correlation.observation_ids
                })

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
                        correlation.correlation_id
                        for correlation in component
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
    ) -> None:
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
