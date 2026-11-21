"""Stance-neutral findings and alternative-aware evaluation definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from samaritan.analysis_contracts import EvidenceReference
from samaritan.contracts import SCHEMA_VERSION, normalize_timestamp, validate_id, validate_json_object, validate_text
from samaritan.proposition_contracts import Proposition, StructuredHypothesis


FINDING_STATUSES = frozenset({"matched", "not_matched", "insufficient"})
CONSISTENCY_JUDGMENTS = frozenset({"expected", "compatible", "unexpected", "incompatible"})
EVALUABILITY_ROLES = frozenset({"required", "contextual"})
DEFINITION_REVIEW_STATES = frozenset({"draft", "reviewed", "approved", "retired"})
AGGREGATE_STANDINGS = frozenset({"favors_focal", "favors_alternative", "non_discriminating", "conflicting"})
RESIDUAL_ALTERNATIVE_ID = "__residual__"


def _schema(value: int, name: str) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError(f"unsupported {name} schema version")


def _json_object(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a JSON object string")
    validated = validate_json_object(value, name)
    parsed = json.loads(validated)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class CriterionParameterDefinition:
    """Declare one typed parameter accepted by a criterion definition."""

    parameter_name: str
    value_type: str
    required: bool = False
    default_value: str | int | bool | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_name", validate_text(self.parameter_name, "parameter_name"))
        object.__setattr__(self, "value_type", validate_text(self.value_type, "value_type"))
        if self.value_type not in {"string", "integer", "boolean"}:
            raise ValueError("unsupported criterion parameter type")
        if type(self.required) is not bool:
            raise TypeError("required must be a boolean")
        if self.required and self.default_value is not None:
            raise ValueError("required criterion parameters cannot have defaults")
        if self.default_value is not None:
            valid = (
                isinstance(self.default_value, str) if self.value_type == "string"
                else type(self.default_value) is int if self.value_type == "integer"
                else type(self.default_value) is bool
            )
            if not valid:
                raise TypeError("criterion parameter default has the wrong type")
            if self.value_type == "string":
                validate_text(self.default_value, "default_value")
        _schema(self.schema_version, "criterion parameter")


@dataclass(frozen=True, slots=True)
class CriterionDefinition:
    """Define an independently testable evidence condition without a stance."""

    criterion_id: str
    criterion_type: str
    parameters_json: str
    description: str
    independence_group: str
    parameter_definitions: tuple[CriterionParameterDefinition, ...] = ()
    outcome_vocabulary_version: int = 1
    evaluator_version: int = 1
    attacker_influenceable: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("criterion_id", "criterion_type", "description", "independence_group"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        object.__setattr__(self, "parameters_json", _json_object(self.parameters_json, "parameters_json"))
        if not isinstance(self.parameter_definitions, tuple) or not all(
            isinstance(item, CriterionParameterDefinition) for item in self.parameter_definitions
        ):
            raise TypeError("parameter_definitions must be a tuple of CriterionParameterDefinition objects")
        parameter_names = tuple(item.parameter_name for item in self.parameter_definitions)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("criterion parameter names must be unique")
        parameter_values = json.loads(self.parameters_json)
        definitions = {item.parameter_name: item for item in self.parameter_definitions}
        if set(parameter_values) - set(definitions):
            raise ValueError("criterion contains undeclared parameters")
        for name, definition in definitions.items():
            if name not in parameter_values:
                if definition.required:
                    raise ValueError(f"missing required criterion parameter: {name}")
                if definition.default_value is not None:
                    parameter_values[name] = definition.default_value
                continue
            value = parameter_values[name]
            valid = (
                isinstance(value, str) if definition.value_type == "string"
                else type(value) is int if definition.value_type == "integer"
                else type(value) is bool
            )
            if not valid:
                raise ValueError(f"criterion parameter type mismatch: {name}")
            if definition.value_type == "string":
                validate_text(value, f"criterion parameter {name}")
        object.__setattr__(
            self,
            "parameters_json",
            json.dumps(parameter_values, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )
        if type(self.outcome_vocabulary_version) is not int or self.outcome_vocabulary_version < 1:
            raise ValueError("outcome_vocabulary_version must be a positive integer")
        if type(self.evaluator_version) is not int or self.evaluator_version < 1:
            raise ValueError("evaluator_version must be a positive integer")
        if type(self.attacker_influenceable) is not bool:
            raise TypeError("attacker_influenceable must be a boolean")
        _schema(self.schema_version, "criterion definition")


@dataclass(frozen=True, slots=True)
class CriterionOutcomeDefinition:
    """Register one finite outcome code and its meaning for a criterion type."""

    criterion_type: str
    outcome_code: str
    meaning: str
    vocabulary_version: int = 1
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("criterion_type", "outcome_code", "meaning"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        if type(self.vocabulary_version) is not int or self.vocabulary_version < 1:
            raise ValueError("vocabulary_version must be a positive integer")
        _schema(self.schema_version, "criterion outcome definition")


SUPPORTED_EVALUATOR_TYPES = frozenset({
    "event_attribute_equals",
    "event_attribute_present",
    "event_type_count",
    "participant_role_matches_proposition_role",
    "temporal_gap_within",
})


@dataclass(frozen=True, slots=True)
class CriterionEvaluatorDefinition:
    """Declare one finite built-in deterministic evaluator and its typed inputs."""

    criterion_type: str
    evaluator_version: int
    evaluator_type: str
    parameter_definitions: tuple[CriterionParameterDefinition, ...]
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("criterion_type", "evaluator_type", "description"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        if type(self.evaluator_version) is not int or self.evaluator_version < 1:
            raise ValueError("evaluator_version must be a positive integer")
        if self.evaluator_type not in SUPPORTED_EVALUATOR_TYPES:
            raise ValueError("unsupported built-in evaluator type")
        if not isinstance(self.parameter_definitions, tuple) or not all(
            isinstance(item, CriterionParameterDefinition) for item in self.parameter_definitions
        ):
            raise TypeError("parameter_definitions must be a tuple of CriterionParameterDefinition objects")
        names = tuple(item.parameter_name for item in self.parameter_definitions)
        if len(names) != len(set(names)):
            raise ValueError("evaluator parameter names must be unique")
        required_types = {
            "event_attribute_equals": {
                "event_role": {"string"},
                "attribute_name": {"string"},
                "expected_value": {"string", "integer", "boolean"},
            },
            "event_attribute_present": {
                "event_role": {"string"},
                "attribute_name": {"string"},
            },
            "event_type_count": {
                "event_type": {"string"},
                "minimum_count": {"integer"},
            },
            "participant_role_matches_proposition_role": {
                "event_role": {"string"},
                "participant_role": {"string"},
                "entity_role": {"string"},
            },
            "temporal_gap_within": {
                "earlier_event_role": {"string"},
                "later_event_role": {"string"},
                "maximum_seconds": {"integer"},
            },
        }[self.evaluator_type]
        supplied = {item.parameter_name: item for item in self.parameter_definitions}
        if set(supplied) != set(required_types):
            raise ValueError("evaluator parameter definitions do not match its built-in interface")
        for name, allowed_types in required_types.items():
            if supplied[name].value_type not in allowed_types or not supplied[name].required:
                raise ValueError(f"invalid required evaluator parameter: {name}")
        _schema(self.schema_version, "criterion evaluator definition")


class CriterionEvaluatorRegistry:
    """Immutable registry of finite built-in evaluator definitions; no callbacks."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, int], CriterionEvaluatorDefinition] = {}

    def register(self, definition: CriterionEvaluatorDefinition) -> None:
        if not isinstance(definition, CriterionEvaluatorDefinition):
            raise TypeError("definition must be CriterionEvaluatorDefinition")
        key = (definition.criterion_type, definition.evaluator_version)
        if key in self._definitions:
            raise ValueError("criterion evaluator version is already registered")
        self._definitions[key] = definition

    def get(self, criterion_type: str, evaluator_version: int) -> CriterionEvaluatorDefinition:
        if type(evaluator_version) is not int or evaluator_version < 1:
            raise ValueError("evaluator_version must be a positive integer")
        key = (validate_text(criterion_type, "criterion_type"), evaluator_version)
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise KeyError(f"unknown criterion evaluator: {key[0]}@{key[1]}") from exc

    def validate_criterion(self, criterion: CriterionDefinition) -> CriterionEvaluatorDefinition:
        if not isinstance(criterion, CriterionDefinition):
            raise TypeError("criterion must be CriterionDefinition")
        definition = self.get(criterion.criterion_type, criterion.evaluator_version)
        if criterion.parameter_definitions != definition.parameter_definitions:
            raise ValueError("criterion parameter definitions do not match its evaluator")
        values = json.loads(criterion.parameters_json)
        if definition.evaluator_type == "event_type_count" and values["minimum_count"] < 1:
            raise ValueError("minimum_count must be positive")
        if definition.evaluator_type == "temporal_gap_within" and values["maximum_seconds"] < 0:
            raise ValueError("maximum_seconds must not be negative")
        return definition


class CriterionOutcomeRegistry:
    """Finite immutable outcome vocabulary, scoped by criterion type."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, int, str], CriterionOutcomeDefinition] = {}

    def register(self, definition: CriterionOutcomeDefinition) -> None:
        if not isinstance(definition, CriterionOutcomeDefinition):
            raise TypeError("definition must be CriterionOutcomeDefinition")
        key = (definition.criterion_type, definition.vocabulary_version, definition.outcome_code)
        if key in self._definitions:
            raise ValueError("criterion outcome code is already registered")
        self._definitions[key] = definition

    def get(self, criterion_type: str, vocabulary_version: int, outcome_code: str) -> CriterionOutcomeDefinition:
        if type(vocabulary_version) is not int or vocabulary_version < 1:
            raise ValueError("vocabulary_version must be a positive integer")
        key = (validate_text(criterion_type, "criterion_type"), vocabulary_version, validate_text(outcome_code, "outcome_code"))
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise KeyError(f"unregistered criterion outcome: {key[0]}@{key[1]}:{key[2]}") from exc

    def contains(self, criterion_type: str, vocabulary_version: int, outcome_code: str) -> bool:
        return (criterion_type, vocabulary_version, outcome_code) in self._definitions


@dataclass(frozen=True, slots=True)
class CriterionFinding:
    """Reusable factual result of evaluating a criterion against evidence."""

    finding_id: str
    criterion_id: str
    status: str
    outcome_code: str | None
    evidence: tuple[EvidenceReference, ...]
    rationale: str
    missing_information: tuple[str, ...] = ()
    created_at: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", validate_id(self.finding_id, "finding_id"))
        object.__setattr__(self, "criterion_id", validate_text(self.criterion_id, "criterion_id"))
        object.__setattr__(self, "status", validate_text(self.status, "status"))
        if self.status not in FINDING_STATUSES:
            raise ValueError("unsupported criterion finding status")
        if self.status == "insufficient":
            if self.outcome_code is not None:
                raise ValueError("insufficient findings must not have an outcome code")
        else:
            if self.outcome_code is None:
                raise ValueError("matched and not_matched findings require an outcome code")
            object.__setattr__(self, "outcome_code", validate_text(self.outcome_code, "outcome_code"))
        if not isinstance(self.evidence, tuple) or not all(isinstance(item, EvidenceReference) for item in self.evidence):
            raise TypeError("evidence must be a tuple of EvidenceReference objects")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("finding evidence references must be unique")
        if self.status != "insufficient" and not self.evidence:
            raise ValueError("evaluated findings require evidence")
        object.__setattr__(self, "rationale", validate_text(self.rationale, "rationale"))
        if not isinstance(self.missing_information, tuple):
            raise TypeError("missing_information must be a tuple")
        for item in self.missing_information:
            validate_text(item, "missing_information item")
        if self.status == "insufficient" and not self.missing_information:
            raise ValueError("insufficient findings require missing information")
        if self.status != "insufficient" and self.missing_information:
            raise ValueError("evaluated findings cannot contain missing information")
        if self.created_at is not None:
            object.__setattr__(self, "created_at", normalize_timestamp(self.created_at, "created_at"))
        _schema(self.schema_version, "criterion finding")

    def validate_against(self, criterion: CriterionDefinition, outcomes: CriterionOutcomeRegistry) -> None:
        if not isinstance(criterion, CriterionDefinition):
            raise TypeError("criterion must be CriterionDefinition")
        if self.criterion_id != criterion.criterion_id:
            raise ValueError("finding criterion ID does not match its definition")
        if self.outcome_code is not None:
            outcomes.get(criterion.criterion_type, criterion.outcome_vocabulary_version, self.outcome_code)


@dataclass(frozen=True, slots=True)
class HypothesisAlternative:
    """A named competing structured proposition."""

    alternative_id: str
    proposition: Proposition
    description: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "alternative_id", validate_id(self.alternative_id, "alternative_id"))
        if not isinstance(self.proposition, Proposition):
            raise TypeError("proposition must be a Proposition")
        object.__setattr__(self, "description", validate_text(self.description, "description"))
        _schema(self.schema_version, "hypothesis alternative")


@dataclass(frozen=True, slots=True)
class BearingDefinition:
    """Versioned interpretation of one finding outcome against one alternative."""

    criterion_id: str
    outcome_code: str
    alternative_id: str
    focal_judgment: str
    alternative_judgment: str
    evaluability_role: str
    rationale: str
    basis: str
    author: str
    review_status: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("criterion_id", "outcome_code", "alternative_id", "rationale", "basis", "author", "review_status"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        for name in ("focal_judgment", "alternative_judgment"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        if self.focal_judgment not in CONSISTENCY_JUDGMENTS:
            raise ValueError("unsupported focal consistency judgment")
        if self.alternative_judgment not in CONSISTENCY_JUDGMENTS:
            raise ValueError("unsupported alternative consistency judgment")
        if self.evaluability_role not in EVALUABILITY_ROLES:
            raise ValueError("unsupported evaluability role")
        if self.review_status not in DEFINITION_REVIEW_STATES:
            raise ValueError("unsupported bearing review status")
        if self.alternative_id == RESIDUAL_ALTERNATIVE_ID and self.alternative_judgment != "compatible":
            raise ValueError("the implicit residual must remain compatible for every outcome")
        _schema(self.schema_version, "bearing definition")


@dataclass(frozen=True, slots=True)
class HypothesisEvaluationRule:
    """Bind criteria and bearings to one exact proposition semantics identity."""

    evaluation_rule_id: str
    rule_version: int
    proposition_type: str
    semantics_version: int
    semantics_fingerprint: str
    criterion_definitions: tuple[CriterionDefinition, ...]
    bearing_definitions: tuple[BearingDefinition, ...]
    description: str
    review_status: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("evaluation_rule_id", "proposition_type", "description", "review_status"):
            object.__setattr__(self, name, validate_text(getattr(self, name), name))
        if type(self.rule_version) is not int or self.rule_version < 1:
            raise ValueError("rule_version must be a positive integer")
        if type(self.semantics_version) is not int or self.semantics_version < 1:
            raise ValueError("semantics_version must be a positive integer")
        object.__setattr__(
            self,
            "semantics_fingerprint",
            validate_text(self.semantics_fingerprint, "semantics_fingerprint"),
        )
        if self.review_status not in DEFINITION_REVIEW_STATES:
            raise ValueError("unsupported evaluation rule review status")
        if not isinstance(self.criterion_definitions, tuple) or not self.criterion_definitions:
            raise TypeError("criterion_definitions must be a non-empty tuple")
        if not all(isinstance(item, CriterionDefinition) for item in self.criterion_definitions):
            raise TypeError("criterion_definitions must contain CriterionDefinition objects")
        if not isinstance(self.bearing_definitions, tuple) or not self.bearing_definitions:
            raise TypeError("bearing_definitions must be a non-empty tuple")
        if not all(isinstance(item, BearingDefinition) for item in self.bearing_definitions):
            raise TypeError("bearing_definitions must contain BearingDefinition objects")
        ids = tuple(item.criterion_id for item in self.criterion_definitions)
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs in an evaluation rule must be unique")
        known = set(ids)
        independence_groups = tuple(item.independence_group for item in self.criterion_definitions)
        if len(independence_groups) != len(set(independence_groups)):
            raise ValueError("evaluation criteria must have distinct independence groups")
        for bearing in self.bearing_definitions:
            if bearing.criterion_id not in known:
                raise ValueError("bearing references an undeclared criterion")
            if self.review_status == "approved" and bearing.review_status != "approved":
                raise ValueError("approved rules require approved bearing definitions")
        focal_judgments: dict[tuple[str, str], str] = {}
        for bearing in self.bearing_definitions:
            key = (bearing.criterion_id, bearing.outcome_code)
            previous = focal_judgments.setdefault(key, bearing.focal_judgment)
            if previous != bearing.focal_judgment:
                raise ValueError(
                    "focal judgment must be consistent across declared alternatives "
                    "for each criterion outcome"
                )
        keys = tuple((item.criterion_id, item.outcome_code, item.alternative_id) for item in self.bearing_definitions)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate bearing definition")
        _schema(self.schema_version, "hypothesis evaluation rule")

    def validate_for(self, proposition: Proposition, production: bool = True) -> None:
        if not isinstance(proposition, Proposition):
            raise TypeError("proposition must be a Proposition")
        if (
            self.proposition_type,
            self.semantics_version,
            self.semantics_fingerprint,
        ) != (
            proposition.proposition_type,
            proposition.semantics_version,
            proposition.semantics_fingerprint,
        ):
            raise ValueError("evaluation rule does not match exact proposition semantics")
        if self.review_status == "retired":
            raise ValueError("retired evaluation rules cannot be used")
        if production and self.review_status != "approved":
            raise ValueError("production evaluation requires an approved rule")
        if not production and self.review_status not in {"reviewed", "approved"}:
            raise ValueError("experimental evaluation requires a reviewed or approved rule")


class HypothesisEvaluationRuleRegistry:
    """Preserve immutable evaluation rule definitions by exact ID and version."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, int], HypothesisEvaluationRule] = {}

    def register(self, definition: HypothesisEvaluationRule) -> None:
        if not isinstance(definition, HypothesisEvaluationRule):
            raise TypeError("definition must be HypothesisEvaluationRule")
        key = (definition.evaluation_rule_id, definition.rule_version)
        if key in self._definitions:
            raise ValueError("evaluation rule version is already registered")
        self._definitions[key] = definition

    def get(self, evaluation_rule_id: str, rule_version: int) -> HypothesisEvaluationRule:
        if type(rule_version) is not int or rule_version < 1:
            raise ValueError("rule_version must be a positive integer")
        key = (validate_text(evaluation_rule_id, "evaluation_rule_id"), rule_version)
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise KeyError(f"unknown evaluation rule: {key[0]}@{key[1]}") from exc

    def __len__(self) -> int:
        return len(self._definitions)


@dataclass(frozen=True, slots=True)
class HypothesisSpecification:
    """A structured hypothesis, declared alternatives and exact evaluation rule."""

    hypothesis: StructuredHypothesis
    alternatives: tuple[HypothesisAlternative, ...]
    evaluation_rule: HypothesisEvaluationRule
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, StructuredHypothesis):
            raise TypeError("hypothesis must be a StructuredHypothesis")
        focal_proposition = self.hypothesis.proposition
        if not isinstance(self.alternatives, tuple) or not self.alternatives:
            raise ValueError("at least one declared alternative is required")
        if not all(isinstance(item, HypothesisAlternative) for item in self.alternatives):
            raise TypeError("alternatives must contain HypothesisAlternative objects")
        ids = tuple(item.alternative_id for item in self.alternatives)
        proposition_ids = tuple(item.proposition.proposition_id for item in self.alternatives)
        if len(ids) != len(set(ids)):
            raise ValueError("alternative IDs must be unique")
        if focal_proposition.proposition_id in proposition_ids:
            raise ValueError("an alternative cannot duplicate the focal proposition")
        if len(proposition_ids) != len(set(proposition_ids)):
            raise ValueError("alternative propositions must be unique")
        if not isinstance(self.evaluation_rule, HypothesisEvaluationRule):
            raise TypeError("evaluation_rule must be HypothesisEvaluationRule")
        for alternative in self.alternatives:
            if (
                alternative.proposition.proposition_type != focal_proposition.proposition_type
                or alternative.proposition.semantics_version != focal_proposition.semantics_version
                or alternative.proposition.semantics_fingerprint
                != focal_proposition.semantics_fingerprint
            ):
                raise ValueError("all alternatives must use the focal proposition semantics")
        allowed_alternative_ids = set(ids) | {RESIDUAL_ALTERNATIVE_ID}
        if any(item.alternative_id not in allowed_alternative_ids for item in self.evaluation_rule.bearing_definitions):
            raise ValueError("evaluation rule contains a bearing for an undeclared alternative")
        self.evaluation_rule.validate_for(focal_proposition, production=False)
        if self.evaluation_rule.review_status == "reviewed" and any(
            item.review_status not in {"reviewed", "approved"}
            for item in self.evaluation_rule.bearing_definitions
        ):
            raise ValueError("reviewed rules require reviewed or approved bearings")
        _schema(self.schema_version, "hypothesis specification")

@dataclass(frozen=True, slots=True)
class AlternativeStanding:
    """Aggregate a set of criterion bearings for one declared alternative."""

    alternative_id: str
    standing: str
    supporting_finding_ids: tuple[str, ...]
    opposing_finding_ids: tuple[str, ...]
    non_discriminating_finding_ids: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "alternative_id", validate_text(self.alternative_id, "alternative_id"))
        if self.standing not in AGGREGATE_STANDINGS:
            raise ValueError("unsupported alternative standing")
        for name in ("supporting_finding_ids", "opposing_finding_ids", "non_discriminating_finding_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            normalized = tuple(validate_id(item, name) for item in values)
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, normalized)
        _schema(self.schema_version, "alternative standing")


@dataclass(frozen=True, slots=True)
class DiagnosticitySynthesis:
    """Deterministic, non-probabilistic result of alternative-aware synthesis."""

    outcome: str
    confidence_ceiling: str
    alternative_standings: tuple[AlternativeStanding, ...]
    reason_codes: tuple[str, ...]
    summary: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outcome not in {"supported", "not_supported", "inconclusive"}:
            raise ValueError("unsupported diagnosticity outcome")
        if self.confidence_ceiling not in {"low", "moderate", "high"}:
            raise ValueError("unsupported confidence ceiling")
        if not isinstance(self.alternative_standings, tuple) or not all(
            isinstance(item, AlternativeStanding) for item in self.alternative_standings
        ):
            raise TypeError("alternative_standings must be a tuple of AlternativeStanding objects")
        if not isinstance(self.reason_codes, tuple):
            raise TypeError("reason_codes must be a tuple")
        for code in self.reason_codes:
            validate_text(code, "reason_code")
        object.__setattr__(self, "summary", validate_text(self.summary, "summary"))
        _schema(self.schema_version, "diagnosticity synthesis")



@dataclass(frozen=True, slots=True)
class HypothesisEvaluationRecord:
    """Preserve the exact rule, findings and synthesis used for one evaluation."""

    evaluation_id: str
    hypothesis_id: str
    evaluation_rule_id: str
    evaluation_rule_version: int
    evidence_set_id: str
    assessment_id: str
    evaluated_at: datetime
    findings: tuple[CriterionFinding, ...]
    synthesis: DiagnosticitySynthesis
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "hypothesis_id", "evidence_set_id", "assessment_id"):
            object.__setattr__(self, name, validate_id(getattr(self, name), name))
        object.__setattr__(
            self,
            "evaluation_rule_id",
            validate_text(self.evaluation_rule_id, "evaluation_rule_id"),
        )
        if type(self.evaluation_rule_version) is not int or self.evaluation_rule_version < 1:
            raise ValueError("evaluation_rule_version must be a positive integer")
        object.__setattr__(self, "evaluated_at", normalize_timestamp(self.evaluated_at, "evaluated_at"))
        if not isinstance(self.findings, tuple) or not all(
            isinstance(item, CriterionFinding) for item in self.findings
        ):
            raise TypeError("findings must be a tuple of CriterionFinding objects")
        criterion_ids = tuple(item.criterion_id for item in self.findings)
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("evaluation findings must have unique criterion IDs")
        if not isinstance(self.synthesis, DiagnosticitySynthesis):
            raise TypeError("synthesis must be a DiagnosticitySynthesis")
        _schema(self.schema_version, "hypothesis evaluation record")

    def validate_against(self, specification: HypothesisSpecification) -> None:
        if not isinstance(specification, HypothesisSpecification):
            raise TypeError("specification must be a HypothesisSpecification")
        if self.hypothesis_id != specification.hypothesis.hypothesis_id:
            raise ValueError("evaluation hypothesis ID does not match specification")
        if (
            self.evaluation_rule_id != specification.evaluation_rule.evaluation_rule_id
            or self.evaluation_rule_version != specification.evaluation_rule.rule_version
        ):
            raise ValueError("evaluation rule identity does not match specification")
        criterion_ids = {
            item.criterion_id for item in specification.evaluation_rule.criterion_definitions
        }
        if any(item.criterion_id not in criterion_ids for item in self.findings):
            raise ValueError("evaluation record contains a finding outside its rule")
        expected_alternatives = {
            item.alternative_id for item in specification.alternatives
        }
        if self.synthesis.confidence_ceiling == "high":
            raise ValueError("M2.4 does not permit a high confidence ceiling")
        if not self.findings and self.synthesis.outcome != "inconclusive":
            raise ValueError(
                "an evaluation with no findings must be inconclusive"
            )
        if self.synthesis.outcome == "supported" and not self.findings:
            raise ValueError("supported evaluations require at least one finding")
        actual_alternatives = {
            item.alternative_id for item in self.synthesis.alternative_standings
        }
        if actual_alternatives != expected_alternatives:
            raise ValueError("evaluation synthesis alternatives do not match specification")
