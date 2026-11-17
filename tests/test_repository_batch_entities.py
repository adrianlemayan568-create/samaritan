
"""Tests for batch entity retrieval in Samaritan's repository."""

import pytest

from samaritan.contracts import new_id
from samaritan.repository import Repository

from test_repository import make_entity, make_observation, make_repository


def persist_entities(repository, source, entities):
    observation = make_observation(source)
    repository.persist_observation(
        observation,
        entities=tuple(entities),
    )


def test_get_entities_by_ids_returns_multiple_entities_in_requested_order(
    tmp_path,
):
    repository, _, source = make_repository(tmp_path)

    first = make_entity("ip_address", "192.0.2.10")
    second = make_entity("user_account", "synthetic-user")
    third = make_entity("hostname", "workstation.example.test")

    persist_entities(repository, source, (first, second, third))

    retrieved = repository.get_entities_by_ids(
        (third.entity_id, first.entity_id, second.entity_id)
    )

    assert retrieved == (third, first, second)


def test_get_entities_by_ids_removes_duplicate_requested_ids(tmp_path):
    repository, _, source = make_repository(tmp_path)

    first = make_entity("ip_address", "192.0.2.10")
    second = make_entity("user_account", "synthetic-user")

    persist_entities(repository, source, (first, second))

    retrieved = repository.get_entities_by_ids(
        (
            first.entity_id,
            second.entity_id,
            first.entity_id,
            second.entity_id,
        )
    )

    assert retrieved == (first, second)


def test_get_entities_by_ids_omits_missing_ids(tmp_path):
    repository, _, source = make_repository(tmp_path)

    existing = make_entity()
    persist_entities(repository, source, (existing,))

    retrieved = repository.get_entities_by_ids(
        (new_id(), existing.entity_id, new_id())
    )

    assert retrieved == (existing,)


def test_get_entities_by_ids_returns_empty_tuple_for_empty_input(
    tmp_path,
):
    repository, _, _ = make_repository(tmp_path)

    assert repository.get_entities_by_ids(()) == ()


def test_get_entities_by_ids_accepts_generator_input(tmp_path):
    repository, _, source = make_repository(tmp_path)

    first = make_entity("ip_address", "192.0.2.10")
    second = make_entity("user_account", "synthetic-user")

    persist_entities(repository, source, (first, second))

    entity_ids = (
        entity_id
        for entity_id in (second.entity_id, first.entity_id)
    )

    assert repository.get_entities_by_ids(entity_ids) == (second, first)


@pytest.mark.parametrize("invalid_input", ["not-an-id-list", b"not-an-id-list", None, 123])
def test_get_entities_by_ids_rejects_non_iterable_or_string_input(
    tmp_path,
    invalid_input,
):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises(TypeError):
        repository.get_entities_by_ids(invalid_input)


@pytest.mark.parametrize("invalid_id", ["", "not-a-uuid", None, 123])
def test_get_entities_by_ids_rejects_invalid_entity_ids(
    tmp_path,
    invalid_id,
):
    repository, _, _ = make_repository(tmp_path)

    with pytest.raises((TypeError, ValueError)):
        repository.get_entities_by_ids((invalid_id,))
