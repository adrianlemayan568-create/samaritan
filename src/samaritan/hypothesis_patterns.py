"""Pure graph matching for evidence-based hypothesis generation."""

from __future__ import annotations

from collections.abc import Iterable

from samaritan.correlation_contracts import Correlation
from samaritan.hypothesis_generation_contracts import (
    HypothesisGenerationRule,
)


def match_correlation_components(
    correlations: Iterable[Correlation],
    rule: HypothesisGenerationRule,
) -> tuple[tuple[Correlation, ...], ...]:
    """Return connected components satisfying a generation rule.

    Only correlations containing at least one required correlation rule ID
    participate in the graph.

    A component qualifies when all required rule IDs occur within it and
    its number of distinct correlation records meets the configured minimum.
    """

    if not isinstance(rule, HypothesisGenerationRule):
        raise TypeError(
            "rule must be a HypothesisGenerationRule"
        )

    if isinstance(correlations, (str, bytes)):
        raise TypeError(
            "correlations must be an iterable of Correlation objects"
        )

    try:
        supplied = tuple(correlations)
    except TypeError as exc:
        raise TypeError(
            "correlations must be an iterable of Correlation objects"
        ) from exc

    if any(
        not isinstance(item, Correlation)
        for item in supplied
    ):
        raise TypeError(
            "all items must be Correlation objects"
        )

    correlation_ids = [
        item.correlation_id for item in supplied
    ]

    if len(correlation_ids) != len(set(correlation_ids)):
        raise ValueError("duplicate correlation IDs are not allowed")

    event_pairs = [
        frozenset(item.event_ids) for item in supplied
    ]

    if len(event_pairs) != len(set(event_pairs)):
        raise ValueError(
            "duplicate unordered event pairs are not allowed"
        )

    required_rule_ids = set(
        rule.required_correlation_rule_ids
    )

    qualifying_edges = tuple(
        item
        for item in supplied
        if any(
            matched.rule_id in required_rule_ids
            for matched in item.matched_rules
        )
    )

    if not qualifying_edges:
        return ()

    adjacency: dict[str, set[str]] = {}

    for correlation in qualifying_edges:
        first, second = correlation.event_ids

        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    visited: set[str] = set()
    components: list[tuple[Correlation, ...]] = []

    for start in sorted(adjacency):
        if start in visited:
            continue

        pending = [start]
        component_events: set[str] = set()

        while pending:
            current = pending.pop()

            if current in visited:
                continue

            visited.add(current)
            component_events.add(current)

            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in visited:
                    pending.append(neighbor)

        component_edges = tuple(
            sorted(
                (
                    correlation
                    for correlation in qualifying_edges
                    if set(correlation.event_ids).issubset(
                        component_events
                    )
                ),
                key=lambda item: item.correlation_id,
            )
        )

        component_rule_ids = {
            matched.rule_id
            for correlation in component_edges
            for matched in correlation.matched_rules
        }

        if (
            required_rule_ids.issubset(component_rule_ids)
            and len(component_edges) >= rule.minimum_correlations
        ):
            components.append(component_edges)

    return tuple(
        sorted(
            components,
            key=lambda component: tuple(
                sorted({
                    event_id
                    for correlation in component
                    for event_id in correlation.event_ids
                })
            ),
        )
    )
