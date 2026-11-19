"""Tests for deterministic canonical identities."""

import hashlib

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from samaritan.identity import (
    CanonicalIdentity,
    IDENTITY_NAMESPACE,
    canonicalize,
    create_canonical_identity,
    verify_canonical_identity,
)


class TestCanonicalize:

    def test_dictionary_order_does_not_change_canonical_bytes(self):
        first = {"account": "A", "event": "login"}
        second = {"event": "login", "account": "A"}

        assert canonicalize(first) == canonicalize(second)

    def test_nested_dictionary_order_does_not_change_canonical_bytes(self):
        first = {"account": {"name": "A", "type": "user"}}
        second = {"account": {"type": "user", "name": "A"}}

        assert canonicalize(first) == canonicalize(second)

    def test_set_order_does_not_change_canonical_bytes(self):
        first = {"roles": {"source", "target", "observer"}}
        second = {"roles": {"observer", "source", "target"}}

        assert canonicalize(first) == canonicalize(second)

    def test_list_order_changes_canonical_bytes(self):
        first = {"sequence": ["login", "logout"]}
        second = {"sequence": ["logout", "login"]}

        assert canonicalize(first) != canonicalize(second)

    def test_integer_is_supported(self):
        assert canonicalize({"count": 5}) == b'{"count":5}'

    def test_float_is_rejected(self):
        with pytest.raises(TypeError):
            canonicalize({"value": 1.5})

    def test_decimal_is_supported(self):
        assert canonicalize({"value": Decimal("1.500")}) == (
            b'{"value":{"$samaritan_type":"decimal","digits":"15","exponent":-1,"sign":0}}'
        )

    def test_equivalent_decimal_values_have_same_canonical_bytes(self):
        first = canonicalize({"value": Decimal("1.50")})
        second = canonicalize({"value": Decimal("1.500")})

        assert first == second

    def test_non_finite_decimal_is_rejected(self):
        with pytest.raises(ValueError):
            canonicalize({"value": Decimal("NaN")})

    def test_aware_datetime_is_normalized_to_utc(self):
        first = datetime(
            2026, 11, 19, 12, 0, tzinfo=timezone.utc
        )
        second = datetime(
            2026, 11, 19, 15, 0,
            tzinfo=timezone(timedelta(hours=3)),
        )

        assert canonicalize({"time": first}) == canonicalize(
            {"time": second}
        )

    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError):
            canonicalize({"time": datetime(2026, 11, 19, 12, 0)})

    def test_non_string_mapping_key_is_rejected(self):
        with pytest.raises(TypeError):
            canonicalize({1: "value"})

    def test_reserved_mapping_key_is_rejected(self):
        with pytest.raises(ValueError):
            canonicalize({"$samaritan_type": "custom"})

    def test_unsupported_value_is_rejected(self):
        with pytest.raises(TypeError):
            canonicalize({"value": object()})

    def test_non_dictionary_input_is_rejected(self):
        with pytest.raises(TypeError):
            canonicalize([1, 2, 3])


class TestCanonicalIdentity:

    def test_identical_material_produces_identical_identity(self):
        first = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )
        second = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )

        assert first == second

    def test_different_object_kinds_produce_different_hashes(self):
        first = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )
        second = create_canonical_identity(
            "evaluation_rule", {"subject": "account-A"}
        )

        assert first.identity_hash != second.identity_hash

    def test_unsupported_canonicalization_version_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported canonicalization"):
            create_canonical_identity(
                "proposition", {"subject": "account-A"}, 2
            )

    def test_historical_version_hash_can_be_verified(self):
        object_kind = "proposition"
        version = 2
        canonical_bytes = b'{"subject":"account-A"}'
        kind_bytes = object_kind.encode("utf-8")

        payload = (
            IDENTITY_NAMESPACE
            + version.to_bytes(4, "big")
            + len(kind_bytes).to_bytes(4, "big")
            + kind_bytes
            + canonical_bytes
        )

        identity = CanonicalIdentity(
            object_kind=object_kind,
            canonicalization_version=version,
            canonical_bytes=canonical_bytes,
            identity_hash=hashlib.sha256(payload).hexdigest(),
        )

        assert verify_canonical_identity(identity) is True

    def test_valid_identity_verifies(self):
        identity = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )

        assert verify_canonical_identity(identity) is True

    def test_tampered_hash_fails_verification(self):
        identity = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )
        tampered = replace(identity, identity_hash="0" * 64)

        assert verify_canonical_identity(tampered) is False

    def test_tampered_canonical_bytes_fail_verification(self):
        identity = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )
        tampered = replace(
            identity,
            canonical_bytes=b'{"subject":"account-B"}',
        )

        assert verify_canonical_identity(tampered) is False

    def test_identity_is_immutable(self):
        identity = create_canonical_identity(
            "proposition", {"subject": "account-A"}
        )

        with pytest.raises(FrozenInstanceError):
            identity.object_kind = "evaluation_rule"

    def test_invalid_object_kind_is_rejected(self):
        with pytest.raises(ValueError):
            create_canonical_identity("", {"subject": "account-A"})

    def test_boolean_version_is_rejected(self):
        with pytest.raises(ValueError):
            create_canonical_identity(
                "proposition", {"subject": "account-A"}, True
            )

    def test_non_positive_version_is_rejected(self):
        with pytest.raises(ValueError):
            create_canonical_identity(
                "proposition", {"subject": "account-A"}, 0
            )

    def test_invalid_identity_hash_is_rejected(self):
        with pytest.raises(ValueError):
            CanonicalIdentity(
                object_kind="proposition",
                canonicalization_version=1,
                canonical_bytes=b'{}',
                identity_hash="invalid",
            )

    def test_non_identity_verification_input_is_rejected(self):
        with pytest.raises(TypeError):
            verify_canonical_identity("not an identity")

    def test_set_and_list_have_different_canonical_bytes(self):
        assert canonicalize({"values": {1, 2}}) != canonicalize(
            {"values": [1, 2]}
        )

    def test_tuple_and_list_have_different_canonical_bytes(self):
        assert canonicalize({"values": (1, 2)}) != canonicalize(
            {"values": [1, 2]}
        )

    def test_frozenset_and_set_have_different_canonical_bytes(self):
        assert canonicalize({"values": frozenset({1, 2})}) != (
            canonicalize({"values": {1, 2}})
        )

    def test_large_decimal_exponent_is_encoded_compactly(self):
        first = canonicalize({"value": Decimal("1E+1000000")})
        second = canonicalize({"value": Decimal("10E+999999")})

        assert first == second
        assert len(first) < 200


class TestCanonicalizationInputHardening:

    def test_unpaired_unicode_surrogate_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be canonicalized"):
            canonicalize({"value": "\ud800"})

    def test_recursive_dictionary_is_rejected(self):
        recursive = {}
        recursive["self"] = recursive

        with pytest.raises(ValueError, match="cannot be canonicalized"):
            canonicalize(recursive)

    def test_recursive_list_is_rejected(self):
        recursive = []
        recursive.append(recursive)

        with pytest.raises(ValueError, match="cannot be canonicalized"):
            canonicalize({"values": recursive})
