# Silver quality report

All figures in this report are computed on **synthetic** Credara data
(fictional client: Credara Fintech).
Cleaning is measured against `data/ground_truth/` and `data/raw/_injected_issues.csv`.

Generated at warehouse `/workspace/portfolio/messy-client-decisioning/data/warehouse.duckdb`.

## Per-issue-type results

A row is **detected** when silver handled the issue correctly, even if the
original ground-truth value could not be restored. That includes quarantining
the raw version, standardising a field so it matches ground truth
(**recovered**), turning a disguised null into a real `NULL`
(**correctly nullified**), or parsing a mixed date onto the correct calendar
day (**day-correct**). **Recall** is detected / injected.
**Recovery rate** is recovered / injected: the share of injections where the
ground-truth value itself could be restored (for example because an earlier
clean version of the row survived).

| issue_type | injected | recovered | quarantined | detected | missed | recall | recovery_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exact_duplicate` | 6119 | 5844 | 275 | 6119 | 0 | 100.0% | 95.5% |
| `near_duplicate` | 478 | 462 | 16 | 478 | 0 | 100.0% | 96.7% |
| `malformed_cpf` | 2400 | 2377 | 0 | 2377 | 23 | 99.0% | 99.0% |
| `mixed_date_format` | 232192 | 128880 | 12069 | 232192 | 0 | 100.0% | 55.5% |
| `currency_as_text` | 376690 | 357338 | 19197 | 376535 | 155 | 100.0% | 94.9% |
| `disguised_null` | 12242 | 845 | 10619 | 12189 | 53 | 99.6% | 6.9% |
| `inconsistent_category` | 42000 | 40787 | 1181 | 41968 | 32 | 99.9% | 97.1% |
| `late_arriving_event` | 10219 | 9661 | 558 | 10219 | 0 | 100.0% | 94.5% |
| `schema_change` | 13 | 13 | 0 | 13 | 0 | 100.0% | 100.0% |
| `outlier` | 180 | 0 | 179 | 179 | 1 | 99.4% | 0.0% |
| `orphan_fk` | 20 | 0 | 20 | 20 | 0 | 100.0% | 0.0% |

### `mixed_date_format` — day-level vs time-of-day

Many injected timestamp formats are date-only strings (for example
`dd/mm/yyyy`) and literally do not carry a time of day. Silver can parse the
calendar day and still cannot restore the original time. Date columns
(`birth_date`, `due_date`) are compared on the calendar day only.

- Injected: 232192
- Day-correct (calendar day matches ground truth): 220123
- Time-of-day lost (day-correct but the timestamp time does not match): 91243
- Fully recovered (exact timestamp, or exact date for date columns): 128880
- Quarantined (unparseable): 12069
- Missed (day wrong, or the entity is absent and not quarantined): 0
- Detection recall: 100.0%
- Recovery rate: 55.5%

### `disguised_null` — detection vs recoverability

A disguised null that silver correctly stores as a real `NULL` is **detected**
but not **recovered**: the original value was overwritten in the raw export and
cannot be invented back. Recovery happens only when a clean earlier or later
version of the same key is the one that survives dedup. Required-field
disguised nulls are quarantined (also detected).

- Injected: 12242
- Recovered (ground-truth value restored): 845
- Correctly nullified (silver is `NULL`): 725
- Quarantined: 10619
- Missed: 53
- Detection recall: 99.6%
- Recovery rate: 6.9%

## Row-level reconciliation (silver vs ground truth)

### applicants

- Ground truth rows: 20000
- Silver rows: 19760
- Matched keys: 19760
- Ground-truth keys missing from silver: 240
- Silver keys not in ground truth: 0
- Key coverage: 98.80%
- date exact-match birth_date: 100.00%
- money exact-match monthly_income: 100.00%

### applications

- Ground truth rows: 50000
- Silver rows: 48200
- Matched keys: 48200
- Ground-truth keys missing from silver: 1800
- Silver keys not in ground truth: 0
- Key coverage: 96.40%
- date exact-match applied_at: 79.94%
- date day-level match applied_at: 100.00%
- date time-of-day lost applied_at: 20.06%
- date exact-match decided_at: 79.96%
- date day-level match decided_at: 100.00%
- date time-of-day lost decided_at: 20.04%
- money exact-match requested_amount: 100.00%
- money exact-match existing_debt: 99.30%

### loans

- Ground truth rows: 30525
- Silver rows: 29418
- Matched keys: 29418
- Ground-truth keys missing from silver: 1107
- Silver keys not in ground truth: 0
- Key coverage: 96.37%
- date exact-match disbursed_at: 79.99%
- date day-level match disbursed_at: 100.00%
- date time-of-day lost disbursed_at: 20.01%
- money exact-match principal: 100.00%
- money exact-match installment_amount: 100.00%

### installments

- Ground truth rows: 558324
- Silver rows: 479362
- Matched keys: 479362
- Ground-truth keys missing from silver: 78962
- Silver keys not in ground truth: 0
- Key coverage: 85.86%
- date exact-match due_date: 100.00%
- date exact-match paid_at: 86.23%
- date day-level match paid_at: 100.00%
- date time-of-day lost paid_at: 13.77%
- money exact-match amount_due: 100.00%
- money exact-match amount_paid: 100.00%

## Why installment coverage is not 100%

Ground truth holds the **full contractual schedule**, including installments
that are not yet due as of the last client extract. Silver can only contain
rows that appeared in a raw batch or that were quarantined from one.
The last batch cutoff / observation date is 2026-10-31 00:00:00+00:00. The generator only exports
installments with `due_date <=` that cutoff, which is realistic for a
snapshot extract.

- Ground-truth installments: 558324
- Silver installments: 479362
- Missing from silver: 78962
- Coverage: 85.9%
- Not present in any raw extract (due after the last cutoff): 51208
- parent loan is in silver (future schedule only): 49207
- parent loan also missing (future schedule and cascade): 2001
- Present in raw but quarantined as `orphan_fk` because the parent loan was rejected: 18822
- Parent loan quarantined, all missing installments: 20823
- Present in raw with a parent loan in silver, but not kept in silver (quarantined as inconsistent_payment or another row rule): 8932

In plain words: most of the gap versus the full ground-truth schedule is
**future installments the client has not yet exported** (due after
2026-10-31). A further share is **cascade from quarantined parents**: if an
applicant, application, or loan fails a silver rule, child installments are
quarantined as orphans. A smaller share is **inconsistent_payment**: a raw
row with `paid_at` filled and `amount_paid` missing is quarantined so a
previous clean version can survive; if none exists, the installment is absent.
Valid, already-exported, consistent installments are not dropped silently.

## Realised default rate (silver vs ground truth)

The label uses the same rule as Analytics Engineering `fct_loan_outcomes`:
a loan is in default if any installment due on or before
`cast(disbursed_at as date) + interval 12 month` reaches 90+ DPD, computed
from `due_date` and `paid_at` against 2026-10-31 UTC. A loan is observable
when that window plus 90 days is on or before the cutoff. All calendar dates
are UTC (`SET TimeZone = 'UTC'`). Host local time (America/Sao_Paulo) is not
used, because it would shift 00:00–02:59 UTC timestamps to the previous day
and can move loans across the observability boundary.

| source | loans | observable | immature | defaults | default rate |
|---|---:|---:|---:|---:|---:|
| silver (contract rule) | 29418 | 26951 | 2467 | 2930 | 10.87% |
| ground truth (same rule) | 30525 | 27972 | 2553 | 3032 | 10.84% |

The loan counts differ because some loans never reach silver (quarantine or
parent cascade). On the same rule and UTC dates, the rates should be close;
a large gap is a cleaning bug, not a definition mismatch.

## Paid-in-full reconciliation (matched installments)

An installment is paid in full when `paid_at` is not null and
`amount_paid >= amount_due`. Silver must not store `amount_paid = 0` (or null)
alongside a filled `paid_at`.

- Matched keys: 479362
- Paid in full in silver: 439418 (91.7%)
- Paid in full in ground truth: 439418 (91.7%)
- Ground-truth paid-in-full but not in silver: 0
- Silver paid-in-full but not in ground truth: 0
- `paid_at` present with zero or null `amount_paid`: 0

## Notable misses and caveats

- `malformed_cpf`: 23 of 2400 injected rows were not detected (1.0% miss rate). Recovered 2377 (99.0%).
- `currency_as_text`: 155 of 376690 injected rows were not detected (0.0% miss rate). Recovered 357338 (94.9%).
- `disguised_null`: 53 of 12242 injected rows were not detected (0.4% miss rate). Recovered 845 (6.9%).
- `inconsistent_category`: 32 of 42000 injected rows were not detected (0.1% miss rate). Recovered 40787 (97.1%).
- `outlier`: 1 of 180 injected rows were not detected (0.6% miss rate). Recovered 0 (0.0%).
- applicants: 240 ground-truth keys are absent from silver. For applicants, applications and loans the usual cause is an outlier or required-field failure on every raw version, or a cascade from a quarantined parent. For installments, see the coverage section: most absences are future dues not present in the raw extracts.
- applications: 1800 ground-truth keys are absent from silver. For applicants, applications and loans the usual cause is an outlier or required-field failure on every raw version, or a cascade from a quarantined parent. For installments, see the coverage section: most absences are future dues not present in the raw extracts.
- loans: 1107 ground-truth keys are absent from silver. For applicants, applications and loans the usual cause is an outlier or required-field failure on every raw version, or a cascade from a quarantined parent. For installments, see the coverage section: most absences are future dues not present in the raw extracts.
- installments: 78962 ground-truth keys are absent from silver. For applicants, applications and loans the usual cause is an outlier or required-field failure on every raw version, or a cascade from a quarantined parent. For installments, see the coverage section: most absences are future dues not present in the raw extracts.
- `stg_quarantine` contains 31212 rejected raw rows (all versions that failed a rule).
- Near-duplicates were injected on a later batch. Silver keeps the latest conforming `ingested_at`. Trailing-space name/channel mutations are trimmed and should match ground truth; a later numeric drift (for example amount + 1) is kept and is counted as not recovered if it disagrees with ground truth.
- `existing_debt` is left `NULL` when the surviving raw value is missing or a disguised null; `existing_debt_is_missing` is true on those rows. A missing balance is never imputed as 0. `amount_paid` is 0 only when the installment is unpaid (`paid_at` is null). A paid_at with a missing amount is quarantined as inconsistent_payment so an earlier clean version can survive.
- A small number of field-level misses may be keys that were quarantined under a different rule (for example an outlier) and are therefore not attributed to the injected issue type.
- Late-arriving installments are expected to appear in silver once the later batch is loaded; they are not quarantined solely for arriving late.

This report does not use data from any real lender.
