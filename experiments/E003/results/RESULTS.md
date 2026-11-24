# E003 — Evidence Integrity Under Duplication, Conflict and Time Anomalies

> **Status: reconstructed experiment record.** This report documents a reconstruction of the E003 test artifact and its execution. The original E003 test file and original results report were not recoverable from the surviving checkout or repository history. This document is not represented as the recovered original report.

## 1. Experiment status

E003 examines how the current ingestion, persistence, correlation, hypothesis-generation, and proposition-construction paths behave under five controlled synthetic evidence conditions:

- A. Cross-observation event-ID collision and transaction rollback.
- B. Distinct record identities with identical content, compared with a controlled distinct-content condition.
- C. Conflicting normalized attributes for records describing the same apparent event time.
- D. An event occurrence timestamp later than its receipt timestamp.
- E. A delayed historical event received after newer events.

The reconstructed experiment is test-only. It adds no production behavior, automatic deduplication, conflict resolution, timestamp correction, or new analytical rules. All inputs are synthetic.

The reconstructed test source is `tests/test_e003_evidence_integrity.py`. It was authored against the E003 protocol and surviving project implementation/tests, not recovered byte-for-byte from the lost original.

## 2. Environment and reproducibility qualification

Execution environment:

- Operating system: antiX 26.1 (Debian GNU/Linux 13.7 base).
- Hardware: IBM desktop with Intel Pentium 4 3.00 GHz; one physical core with Hyper-Threading.
- Memory: approximately 957 MiB RAM; 2 GiB swap.
- Python: 3.13.5.
- pytest: 9.2.0.
- Project declaration: `requires-python = ">=3.14"`.

**Compatibility qualification:** the available IBM environment runs Python 3.13.5, below the project's declared minimum of Python 3.14. These successful runs demonstrate compatibility in the available environment; they do not establish supported-runtime validation. A run under Python 3.14 or later remains necessary.

The experiment uses temporary SQLite databases and synthetic records. The test suite is deterministic in its inputs and fixed experimental timestamps. No external or real-world evidence was used.

## 3. Research questions and case results

### Case A — Cross-observation event-ID collision

**Question:** If a second observation attempts to persist an event using an event ID already persisted by an earlier observation, does the failed persistence leave a partial observation or alter the original event?

**Method:** Persist an initial synthetic event. Submit a second observation with the same event ID and different normalized content. Expect the SQLite uniqueness constraint to reject the second write.

**Observed result:** the second ingestion raises `sqlite3.IntegrityError`. The reconstructed test verifies that only the first observation remains, the original observation is unchanged, the original event is unchanged, and the event remains associated with the original observation.

**Interpretation:** the tested persistence path is transactional for this collision: the failed second write does not leave a partially persisted observation or replace the original event.

**Scope limitation:** this establishes behavior for the tested repository transaction and event-ID collision path. It is not a proof that every possible persistence failure mode has identical rollback behavior.

### Case B — Distinct record identity versus content equality

**Question:** Can separately identified records with identical normalized content contribute separately to correlation and hypothesis generation? Does changing content alone alter that result?

**Method:** Run three isolated synthetic scenarios:

1. Baseline: one authentication-failure event, one success event, and one host-activity event.
2. Duplicate-content condition: two separately identified authentication-failure records with identical normalized content, followed by the same success and host-activity pattern.
3. Controlled distinct-content condition: the same two failure records at the same occurrence time, with the same event type and participant context, but with one normalized data field changed in the second record.

The controlled comparison holds the timestamp and participant context constant; the normalized content field is the intended difference between the duplicate and distinct-content conditions.

**Observed result:**

| Scenario | Correlations | Generated hypotheses |
|---|---:|---:|
| Baseline | 2 | 0 |
| Duplicate content | 3 | 1 |
| Distinct-content control | 3 | 1 |

The two duplicate-content failure records have distinct event IDs but identical normalized data and occurrence timestamps. The distinct-content control also has distinct event IDs and matching occurrence timestamps, but the normalized data differs. Both two-failure scenarios produce the same correlation and hypothesis counts.

**Interpretation:** record identity is distinct from content equality. In the tested correlation path, separately identified records can contribute additional graph edges even when their normalized content is identical. The controlled comparison is consistent with the inspected correlation matcher: it uses event types, shared participant entities, and event-time distance; normalized event content is not a matching input.

This does **not** establish that the two records represent two independent real-world occurrences. Nor does identical content prove that they represent one occurrence. The experiment supplies no defensible basis for automatic deduplication or for treating record identity as evidence independence.

**Scope limitation:** the result is specific to the configured synthetic rules and the tested correlation/generation path. The experiment does not establish general occurrence identity or evidential independence.

### Case C — Conflicting normalized attributes

**Question:** When separately persisted records have the same event type and occurrence time but conflicting normalized attributes, are both records and their provenance retained, and does proposition construction resolve the conflict arbitrarily?

**Method:** Ingest a synthetic authentication sequence, then ingest a second authentication-success event for the same account at the same occurrence time with a conflicting normalized result attribute. Construct evidence containing both records and their observation references.

**Observed result:** both events remain separately persisted with distinct event IDs, the same occurrence timestamp, different normalized content, and their respective observation provenance. Proposition construction raises `PropositionConstructionError` with the specific message `event-role selection is ambiguous`.

**Interpretation:** the tested path preserves the conflicting source records and surfaces ambiguity during event-role selection rather than silently choosing one of the conflicting records.

**Scope limitation:** this is evidence about the exercised proposition-construction role-selection path, not a universal guarantee that every kind of conflict is detected or represented identically.

### Case D — Future occurrence timestamp

**Question:** Does ingestion preserve an event whose occurrence timestamp is later than the observation receipt timestamp, and are the two times kept distinct?

**Method:** Ingest one synthetic authentication-failure event with an occurrence timestamp 60 minutes after the fixed receipt time. Verify the original timestamp string, parsed occurrence timestamp, and observation receipt timestamp independently.

**Observed result:** ingestion accepts the event. The exact original timestamp string is retained; the normalized occurrence timestamp equals the parsed original timestamp; the occurrence time is later than the independently persisted receipt time.

**Interpretation:** the tested ingestion path does not reject this future-dated event and does not collapse occurrence time into receipt time.

**Scope limitation:** the reconstructed Case D test verifies ingestion and timestamp persistence only. It does not test downstream correlation, hypothesis generation, or proposition construction for this future-dated record, and it does not establish why the timestamp is in the future. Clock skew, source-clock error, and other explanations are not distinguished. No automatic timestamp correction is justified by this result.

### Case E — Delayed historical event

**Question:** When an event with an earlier occurrence time arrives after newer events, does the system preserve receipt order separately from event-time order, and can the resulting evidence still pass through the tested analytical path?

**Method:** Ingest a success event, then a host-activity event, then a failure event whose occurrence time predates both but whose receipt time is later. Inspect observation and event retrieval order, then run correlation, hypothesis generation, and proposition construction.

**Observed result:** observations are returned in receipt-time order, while events are returned in occurrence-time order. The delayed event retains an occurrence time earlier than its receipt time. The selected sequence produces two correlations and one hypothesis, and proposition construction succeeds.

**Interpretation:** in the tested default retrieval path, receipt chronology and event chronology remain distinct. A delayed historical event is retained and can participate in the tested downstream sequence.

**Scope limitation:** this result covers the default retrieval methods and the three-event synthetic case. It does not establish ordering behavior for every pagination, filtering, or retrieval path.

## 4. Test execution results

### Focused reconstructed E003 suite

Command:

```bash
PYTHONPATH=src:tests ~/samaritan-test-venv/bin/python -m pytest -q tests/test_e003_evidence_integrity.py
```

Observed output:

```text
.....                                                                    [100%]
5 passed in 1.14s
```

### Full project regression suite

Command:

```bash
PYTHONPATH=src:tests ~/samaritan-test-venv/bin/python -m pytest -q
```

Observed output:

```text
749 passed in 12.69s
```

Both runs were performed on Python 3.13.5, below the declared Python >=3.14 requirement. The results must be read with the environment qualification in Section 2.

## 5. Consolidated findings

1. **Transactional collision handling:** the tested duplicate event-ID write fails without leaving a second observation or changing the original persisted event.
2. **Identity is not occurrence identity:** distinct event IDs can refer to records with identical content. The tested correlation engine can count both records; this alone does not show that they represent independent occurrences.
3. **Content is not a correlation discriminator in this path:** the controlled distinct-content comparison changes one normalized data field while holding event type, occurrence time, and participant context constant; the correlation and hypothesis counts remain the same.
4. **Conflicts can remain explicit:** the tested conflicting records and their observation provenance are preserved, and proposition construction reports ambiguity for the tested event-role selection.
5. **Occurrence and receipt time are separate:** the tested future-dated event is accepted with both timestamps preserved independently.
6. **Late arrival does not rewrite event chronology:** in the tested default retrieval path, observation order follows receipt time while event order follows occurrence time.

The central distinction is between **record identity**, **real-world occurrence identity**, and **evidence independence**. These are different concepts. The current experiment does not justify inferring one from another.

## 6. Limitations and non-claims

- This is a reconstruction, not the original E003 test source or original report. The original files were not recoverable; the reconstructed test file should not be described as byte-for-byte restoration.
- The tests use synthetic inputs and controlled local SQLite databases only.
- The results describe the exercised code paths and configurations, not all possible ingestion, persistence, retrieval, or analytical behavior.
- The duplicate-content experiment cannot determine whether two records describe one real-world occurrence or two.
- The distinct-content control changes one normalized data field; it does not establish a universal theory of event equivalence.
- Case C exercises one specific proposition-construction ambiguity path.
- Reconstructed Case D checks timestamp ingestion and persistence, not downstream analytical consequences or the cause of the future timestamp.
- Case E uses the default retrieval methods and a small three-event sequence; it is not an exhaustive pagination or ordering test.
- No deduplication, conflict resolution, timestamp correction, automatic trust scoring, or evidence-independence inference was introduced.
- The available runtime is Python 3.13.5 despite the project's Python >=3.14 declaration. Supported-runtime validation remains outstanding.
- Passing tests establish that the implementation satisfies these test assertions in this environment. They do not prove that the assertions exhaustively model real-world evidence behavior or that the system is ready for consequential real-world investigations.

## 7. Conclusion

The reconstructed E003 experiment passes all five focused tests and the full 749-test project suite in the available Python 3.13.5 environment.

Within the tested synthetic paths, Samaritan preserves distinct record identities and provenance, rolls back the tested event-ID collision transaction, surfaces the tested proposition-role ambiguity, keeps occurrence and receipt timestamps distinct, and maintains separate receipt-time and event-time ordering.

The strongest conclusion is not that the system can identify duplicate real-world occurrences or establish independent evidence. It is that **record identity, occurrence identity, and evidence independence must not be conflated**. The experiment supplies no basis for automatic deduplication or for treating separate records as independent corroboration.

E003 remains a controlled software-behavior experiment. It does not validate Samaritan for consequential real-world use. A supported-runtime run under Python 3.14 or later remains outstanding.
