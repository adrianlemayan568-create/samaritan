# E001 — Authentication Sequence Validation

## 1. Experiment identity

- Experiment ID: E001
- Source commit: d715e057bcb260c129b4fb4b97488224939637b2
- Repository branch: m2.4-audit-corrections
- Execution machine: IBM antiX desktop
- Operating environment: Debian GNU/Linux 13 (Trixie), antiX
- Python: 3.13.5
- pytest: 9.2.0
- Date: 2026-11-22
- Dataset: synthetic, controlled, known-ground-truth telemetry
- Scope: authentication sequence construction and evaluation

This experiment was executed from an isolated checkout. It did not modify
the original development checkout or the earlier source snapshot.

## 2. Objective

Test whether the existing analytical pipeline can process controlled
authentication telemetry and produce correctly constrained, traceable
analytical outputs.

The experiment covers ingestion, correlation, hypothesis generation,
EvidenceSet selection, proposition construction, criterion evaluation,
diagnosticity synthesis, and source provenance.

This is a narrow experiment. It is not a general validation of Samaritan
or a test of real-world investigative accuracy.

## 3. Frozen seven-case matrix and observed outcomes

| Case | Scenario | Expected behavior | Result |
|---|---|---|---|
| A | Valid sequence for one account | Construct proposition | PASS |
| B | Events belong to different accounts | Do not combine accounts | PASS |
| C | Failure occurs after success | Reject temporal ordering | PASS |
| D | Required host activity is absent | Reject construction | PASS |
| E | Multiple failures match one required role | Reject ambiguous construction | PASS |
| F | Shared IP, different account identities | Do not infer shared account | PASS |
| G | Required normalized attribute is absent | Return insufficient information | PASS |

Result: 7/7 tests passed.

Case A also checks:
- Ingestion through the actual ingestion service.
- Account-restricted correlation.
- Hypothesis generation and evidence references.
- Proposition construction and provenance.
- Separate focal and alternative EvidenceSets.
- Deterministic proposition and evaluation identities.
- Neutral reviewed bearings.
- Inconclusive synthesis with low confidence.

## 4. Evaluation policy and interpretation

The focal proposition represents a valid authentication sequence for
account A. The declared alternative represents a separately constructed
valid sequence for account B, with its own persisted EvidenceSet.

The alternative comparison uses reviewed, deliberately neutral bearings.
The expected baseline synthesis is therefore:

- Outcome: inconclusive
- Confidence ceiling: low

This is intentional. The experiment must not manufacture a favorable
result through biased evaluation bearings.

Important implementation limitation:

The evaluation engine authenticates the historical construction manifests
for both focal and alternative propositions. However, criterion evaluators
run only against the focal proposition and its EvidenceSet. The alternative
is not independently evaluated against its own EvidenceSet.

The synthesis therefore compares focal findings with pre-authored bearings.
It must not be described as independent alternative evaluation.

The result "inconclusive" is not evidence that the sequence is false.
Likewise, a hypothetical "supported" result would not prove that it is true.

## 5. Test execution results

### Dedicated E001 run

Command:

    PYTHONPATH=src ~/samaritan-test-venv/bin/python -m pytest \
      tests/test_e001_authentication_sequence.py -v

Observed result:

    7 passed in 1.13s

### Full regression suite

Command:

    PYTHONPATH=src ~/samaritan-test-venv/bin/python -m pytest -q

Observed result:

    743 passed in 11.86s

### JUnit report run

Observed result:

    7 passed, 7 warnings in 0.87s

The seven warnings concern pytest record_property compatibility with the
configured xunit2 JUnit family. Inspection confirmed that the generated
XML nevertheless contains the recorded properties.

No test failures were reported.

## 6. Recorded measurements

The following measurements were recorded for Case A in the JUnit report.

| Measurement | Value |
|---|---:|
| Correlation | 0.008503253 s |
| Hypothesis generation | 0.008360983 s |
| Proposition construction | 0.031014402 s |
| Evaluation and synthesis | 0.016401481 s |
| Correlation count | 2 |
| Generated hypothesis count | 1 |
| Recorded event count | 3 |
| Recorded observation count | 2 |

These are single-run measurements from one small synthetic case.
They are observations, not performance benchmarks. They do not establish
scaling behavior, throughput, or performance on larger datasets.

## 7. Raw evidence and report integrity

Raw JUnit report:

    e001-results.xml

SHA-256:

    a768e93a4ef2e77e3305a53d1b3a601c29ca6de1bd7e6418c63c15d5e71f2b92

The copied report checksum was verified against the original report.

## 8. Findings

Within the seven specified synthetic cases, the tested implementation
behaved as expected.

In particular, the experiment observed that:
- Different account identities were not combined into one proposition.
- Shared IP identity did not establish shared account identity.
- Invalid temporal ordering was rejected during construction.
- Missing required events were rejected.
- Ambiguous event-role assignment was rejected.
- Missing required normalized information produced an insufficient
  finding rather than a negative finding.
- A valid focal sequence could be constructed with traceable provenance.
- Neutral evaluation bearings produced an inconclusive, low-confidence
  synthesis rather than an artificially favorable result.

## 9. Limitations

This experiment does not establish:
- General analytical accuracy outside these seven cases.
- Correctness against real-world telemetry distributions.
- Resistance to adversarially manipulated evidence.
- Independent evaluation of alternative-specific evidence.
- Actor identity, authorization, intent, or causation.
- Performance at production scale.
- Readiness for consequential real-world investigations.
- Official Python 3.13 support; the project declares Python >=3.14.
  Python 3.13.5 execution is compatibility evidence for this tested
  checkout, not a change to the declared support policy.

All data used here is synthetic. No real third-party investigative data
was used.

## 10. Disposition

E001: PASSED WITH DOCUMENTED LIMITATIONS.

This means the implementation passed the frozen narrow experiment.
It does not mean Samaritan is validated for general investigative use.

Preserve this experiment, its protocol, source commit, raw report, and
measurements as a reproducible baseline.

No production source changes, merges, or promotion decisions are part
of this result.
