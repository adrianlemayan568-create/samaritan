# E002 — Alternative-Specific Evidence Evaluation

## 1. Identity and scope

- Experiment: E002
- Branch: `m2.4-audit-corrections`
- Baseline commit: `e1ba105`
- Test: `tests/test_e002_alternative_evidence.py`
- Results: `experiments/E002/results/RESULTS.md`
- Environment: IBM desktop, antiX Linux, Python 3.13.5, pytest 9.2.0
- Data: synthetic only
- Implementation: test-only; no production source changes

## 2. Research question

Does changing an alternative proposition's own evidence change the criterion findings used by `HypothesisEvaluationEngine` and its synthesis?

The experiment holds the focal proposition and focal EvidenceSet constant while changing only the normalized authentication result in a separately constructed alternative sequence.

## 3. Controlled design

One focal authentication sequence for account A was ingested, persisted, and constructed. The same focal proposition and EvidenceSet were used in both comparisons.

Two alternative sequences for account B were separately ingested, persisted, and constructed. Each had its own proposition identity and EvidenceSet. Both used the same proposition type, semantics version, construction definition, criterion configuration, and logical alternative identifier.

| Comparison | Focal result | Alternative result |
|---|---|---|
| A | `success` | `failure` |
| B | `success` | `success` |

The same registered criterion evaluator and outcome vocabulary were used for both the independent control evaluations and orchestration.

### Independent evaluator control

The test directly evaluated the criterion against:
- The focal proposition and focal EvidenceSet.
- Each alternative proposition and its own EvidenceSet.

This establishes what the registered evaluator returns when each proposition is evaluated against its own evidence. It does not imply that orchestration performs alternative-specific evaluation.

### Orchestration under test

For each comparison, `HypothesisEvaluationEngine.evaluate()` received the same focal EvidenceSet. The alternative's own construction manifest was authenticated by the normal orchestration path.

The approved neutral experimental bearings were retained. They deliberately do not discriminate between the focal proposition and the declared alternative.

## 4. Observed results

### Independent criterion evaluation

| Evidence evaluated | Alternative A: failure | Alternative B: success |
|---|---|---|
| Focal sequence | `matched` / `equal` | `matched` / `equal` |
| Alternative sequence | `not_matched` / `not_equal` | `matched` / `equal` |

The alternative-specific finding changed with the alternative's normalized result. The test also verified that each finding referenced the appropriate authentication-success event and source observation.

### Orchestration

| Output | Alternative A: failure | Alternative B: success |
|---|---|---|
| Findings returned | Focal finding only | Focal finding only |
| Focal outcome | `equal` | `equal` |
| Synthesis | `inconclusive` | `inconclusive` |
| Confidence ceiling | `low` | `low` |
| Alternative standing | `non_discriminating` | `non_discriminating` |

The orchestration findings were identical across both comparisons despite the different independent alternative findings. The test asserted that the orchestration finding exactly equaled the independently evaluated focal finding. No alternative-specific criterion finding appeared in the orchestration record.

## 5. Interpretation

**Observed implementation limitation:** `HypothesisEvaluationEngine` authenticates the construction evidence manifests of focal and alternative propositions, but its criterion-evaluation loop evaluates criteria only against the focal proposition and focal EvidenceSet.

The resulting focal findings and registered bearings are passed to diagnosticity synthesis. Alternative-specific criterion findings are not represented in the orchestration findings collection and are not incorporated into that synthesis.

This is a demonstrated limitation of the current implementation.

The experiment does not establish what the synthesis outcome should be after alternative-specific evaluation is introduced. Neutral bearings were deliberately retained; no replacement comparative-synthesis policy was tested or approved.

## 6. Test results

- Dedicated E002 test: **1 passed** in 0.55 seconds.
- Full test suite: **744 passed** in 10.54 seconds.
- JUnit report run: **1 passed** in 0.46 seconds, with one warning.
- The warning concerns `record_property` compatibility with the configured `xunit2` JUnit format. The report nevertheless contains the recorded properties.

## 7. Timing observations

Single-run measurements from the JUnit report:

| Operation | Alternative A: failure | Alternative B: success |
|---|---:|---:|
| Independent focal evaluation | 10.494 ms | 10.629 ms |
| Independent alternative evaluation | 10.305 ms | 10.552 ms |
| Orchestration | 16.073 ms | 16.140 ms |

These are experimental observations, not performance benchmarks or capacity claims.

The experiment recorded two comparisons and verified that the focal EvidenceSet remained unchanged. Production source remained unmodified.

## 8. Limitations

- Synthetic data only; no real-world investigative validity is established.
- The alternative is another valid account-authentication sequence, not an independently modeled causal explanation.
- The experiment tests current evaluator and orchestration behavior; it does not validate a future alternative-aware synthesis design.
- Shared account, host, or IP references do not establish human identity, authorization, causation, or malicious intent.
- Python 3.13.5 is below the project's declared Python requirement of `>=3.14`. This run is cross-version compatibility evidence, not official support certification.
- Timing values come from single runs and should not be treated as benchmarks.

## 9. Disposition

E002 confirms a material analytical limitation: alternative construction evidence is authenticated but is not independently evaluated or represented in the current hypothesis-evaluation record.

The result should inform a separate architecture decision before any production change. This experiment itself makes no implementation change and does not approve a replacement evaluation or synthesis policy.

**Status:** Experiment completed; findings documented; no production changes; no merge or promotion.
