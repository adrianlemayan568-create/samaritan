"""Deterministic canonical identities for Samaritan objects."""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from samaritan.contracts import validate_text


CANONICALIZATION_VERSION = 1
SUPPORTED_CANONICALIZATION_VERSIONS = frozenset({1})
IDENTITY_NAMESPACE = b"SAMARITAN-IDENTITY\x00"
_RESERVED_KEY = "$samaritan_type"


def _normalize_value(value: Any) -> Any:
    """Convert supported values into an unambiguous JSON-compatible form."""

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        raise TypeError("floating-point values are not supported in identities")

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("decimal values must be finite")

        sign, digits, exponent = value.as_tuple()

        if not any(digits):
            sign, digits, exponent = 0, (0,), 0
        else:
            digits = list(digits)

            while len(digits) > 1 and digits[-1] == 0:
                digits.pop()
                exponent += 1

            digits = tuple(digits)

        return {
            _RESERVED_KEY: "decimal",
            "sign": sign,
            "digits": "".join(str(digit) for digit in digits),
            "exponent": exponent,
        }

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("identity datetimes must include a timezone")

        normalized = value.astimezone(timezone.utc)
        timestamp = normalized.isoformat(timespec="microseconds")
        timestamp = timestamp.replace("+00:00", "Z")

        return {
            _RESERVED_KEY: "datetime",
            "value": timestamp,
        }

    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("identity mapping keys must be strings")

        if _RESERVED_KEY in value:
            raise ValueError(f"{_RESERVED_KEY} is a reserved mapping key")

        return {
            key: _normalize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_normalize_value(item) for item in value]

    if isinstance(value, tuple):
        return {
            _RESERVED_KEY: "tuple",
            "items": [_normalize_value(item) for item in value],
        }

    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_value(item) for item in value]

        sorted_items = sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )

        collection_type = (
            "frozenset" if isinstance(value, frozenset) else "set"
        )

        return {
            _RESERVED_KEY: collection_type,
            "items": sorted_items,
        }

    raise TypeError(
        f"unsupported identity value type: {type(value).__name__}"
    )


def canonicalize(identity_material: dict[str, Any]) -> bytes:
    """Serialize validated identity material into deterministic UTF-8 bytes."""

    if not isinstance(identity_material, dict):
        raise TypeError("identity_material must be a dictionary")

    try:
        normalized = _normalize_value(identity_material)
    except RecursionError as exc:
        raise ValueError(
            "identity material cannot be canonicalized"
        ) from exc

    try:
        serialized = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(
            "identity material cannot be canonicalized"
        ) from exc

    try:
        return serialized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "identity material cannot be canonicalized"
        ) from exc


def _identity_digest(
    object_kind: str,
    canonicalization_version: int,
    canonical_bytes: bytes,
) -> str:
    """Hash canonical bytes within a versioned object-kind domain."""

    kind_bytes = object_kind.encode("utf-8")

    payload = (
        IDENTITY_NAMESPACE
        + canonicalization_version.to_bytes(4, "big")
        + len(kind_bytes).to_bytes(4, "big")
        + kind_bytes
        + canonical_bytes
    )

    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalIdentity:
    """Immutable canonical identity and its verification material."""

    object_kind: str
    canonicalization_version: int
    canonical_bytes: bytes
    identity_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_kind",
            validate_text(self.object_kind, "object_kind"),
        )

        if (
            not isinstance(self.canonicalization_version, int)
            or isinstance(self.canonicalization_version, bool)
            or not 1 <= self.canonicalization_version <= 0xFFFFFFFF
        ):
            raise ValueError(
                "canonicalization_version must be a positive integer"
            )

        if not isinstance(self.canonical_bytes, bytes):
            raise TypeError("canonical_bytes must be bytes")

        if not isinstance(self.identity_hash, str):
            raise TypeError("identity_hash must be a string")

        if (
            len(self.identity_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.identity_hash)
        ):
            raise ValueError(
                "identity_hash must be a lowercase SHA-256 hex digest"
            )


def create_canonical_identity(
    object_kind: str,
    identity_material: dict[str, Any],
    canonicalization_version: int = CANONICALIZATION_VERSION,
) -> CanonicalIdentity:
    """Create a deterministic, versioned identity for an object."""

    object_kind = validate_text(object_kind, "object_kind")

    if (
        not isinstance(canonicalization_version, int)
        or isinstance(canonicalization_version, bool)
        or canonicalization_version not in SUPPORTED_CANONICALIZATION_VERSIONS
    ):
        raise ValueError(
            "unsupported canonicalization version"
        )

    canonical_bytes = canonicalize(identity_material)

    identity_hash = _identity_digest(
        object_kind,
        canonicalization_version,
        canonical_bytes,
    )

    return CanonicalIdentity(
        object_kind=object_kind,
        canonicalization_version=canonicalization_version,
        canonical_bytes=canonical_bytes,
        identity_hash=identity_hash,
    )


def verify_canonical_identity(identity: CanonicalIdentity) -> bool:
    """Verify that the stored hash matches its canonical identity material."""

    if not isinstance(identity, CanonicalIdentity):
        raise TypeError("identity must be a CanonicalIdentity")

    expected = _identity_digest(
        identity.object_kind,
        identity.canonicalization_version,
        identity.canonical_bytes,
    )

    return expected == identity.identity_hash
