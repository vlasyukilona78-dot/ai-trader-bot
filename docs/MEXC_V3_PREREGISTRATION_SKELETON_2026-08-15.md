# MEXC v3 — preregistration skeleton

> **Status:** `TEMPLATE_ONLY / NOT_FROZEN / NOT_EXECUTABLE`
>
> **Parent authority:** `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md`
>
> **Purpose:** empty, reviewable structure for the P7 preregistration package.
> It is not a completed preregistration, experiment approval, model manifest,
> network authorization, trading instruction, or evidence receipt.

## 0. Hard-block semantics

The sentinel below is deliberate:

```text
__UNFILLED__
```

Every required field that still contains `__UNFILLED__`, an empty checkbox, an
unresolved conflict, or a missing referenced receipt blocks the phase named in
that field and every dependent later phase. `TBD`, an implicit default, an env
value, a notebook cell, or a verbal agreement does not count as completion.

This skeleton records **no parameter values and no result values**. Creating or
reviewing it does not authorize:

- public or private network access;
- a MEXC data pilot or data collection;
- opening `.env` or using any credential;
- scanner, bot, Telegram, private API, testnet, demo, or live runtime;
- dataset construction, holdout inspection, model fitting, threshold search, or
  performance claims;
- actionable entry, quantity, stop, TP, or real-money instruction.

The approved master plan remains authoritative if this empty template differs
from it. A filled preregistration may narrow that plan but may not silently
weaken its causal, evidence, risk, SLA, population, or release invariants.

## 1. Document identity and freeze receipt

| Field | Required value | Required before |
|---|---|---|
| preregistration schema/version | `__UNFILLED__` | P7 acceptance |
| preregistration instance ID | `__UNFILLED__` | P7 acceptance |
| hypothesis family/version ID | `__UNFILLED__` | candidate development |
| parent master-plan path + content hash | `__UNFILLED__` | P7 acceptance |
| repository commit | `__UNFILLED__` | P7 acceptance |
| dirty-worktree policy/receipt | `__UNFILLED__` | P7 acceptance |
| author(s) | `__UNFILLED__` | P7 acceptance |
| independent reviewer(s) | `__UNFILLED__` | P7 acceptance |
| freeze timestamp (UTC) | `__UNFILLED__` | P7 acceptance |
| canonical serialization | `__UNFILLED__` | P7 acceptance |
| canonical payload SHA-256 (`hash` field normalized to `__HASH_SLOT__`) | `__UNFILLED__` | P7 acceptance |
| detached freeze-manifest path + SHA-256 | `__UNFILLED__` | P7 acceptance |
| detached/off-device checkpoint receipt | `__UNFILLED__` | final holdout opening |
| amendment policy/version successor rule | `__UNFILLED__` | candidate development |

Freeze uses an acyclic hash graph. `canonical_payload_sha256` is computed over
the canonical document with its own hash cell replaced by the literal
`__HASH_SLOT__`. A detached `FreezeManifest` binds that payload hash to every
referenced contract, data/split manifest and seed list; the external checkpoint
receipt anchors the freeze-manifest hash. No object embeds the hash of a parent
that already embeds that object's hash. A later semantic change creates a new
hypothesis version; it never edits an opened holdout's history in place.

## 2. Hypothesis and estimands

### 2.1 Primary hypothesis

| Field | Required value |
|---|---|
| one-sentence causal hypothesis | `__UNFILLED__` |
| direction/side and event class | `__UNFILLED__` |
| null hypothesis | `__UNFILLED__` |
| alternative hypothesis | `__UNFILLED__` |
| target market and instrument class | `__UNFILLED__` |
| target point-in-time population | `__UNFILLED__` |
| intervention/policy being evaluated | `__UNFILLED__` |
| primary comparator | `__UNFILLED__` |
| absolute economic estimand | `__UNFILLED__` |
| incremental economic estimand | `__UNFILLED__` |
| safety estimands | `__UNFILLED__` |
| unit of analysis | `__UNFILLED__` |
| calendar/portfolio unit | `__UNFILLED__` |
| claims explicitly outside scope | `__UNFILLED__` |

The hypothesis must be testable from point-in-time evidence. It may not claim
that a detected peak is a future global maximum, that price cannot go higher,
or that a fixed stop fill is guaranteed during a gap.

### 2.2 Version and contract bindings

| Contract/artifact | Version | Content hash | Compatibility fixture/receipt |
|---|---|---|---|
| universe policy | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| raw-data/request schema | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| Min1 aggregation contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| StrategySpecV3 | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| FeatureSnapshotV3 | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| PeakEpisode contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| instrument-spec contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| proposal geometry contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| cost contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| sizing/risk policy | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| fill/execution-stress contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| outcome/label contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| single-position/portfolio replay | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| fast/context/proposal/outcome ledgers | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| baseline evaluation contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| split/purge contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| verdict contract | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Frozen v2/Journal-v6/single-position-v3 artifacts are controls and compatibility
fixtures only. V3 semantics require their own version-dispatched artifacts and
must not be written into the frozen v2 evidence line.

## 3. Point-in-time populations and denominators

The following transitions are mandatory. Exact definitions are intentionally
blank until P3 proves the acquisition policy feasible and P4 freezes the
contracts.

| Population | Inclusion predicate | Exclusions/typed reasons | Denominator source | Required receipt |
|---|---|---|---|---|
| `raw_contract_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `fast_scan_valid_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `pump_formation_eligible_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `context_requested_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `context_complete_or_typed_missing_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `feature_valid_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `proposal_eligible_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `entered_trade_population` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Required population declarations:

| Declaration | Required value |
|---|---|
| point-in-time listing/delisting semantics | `__UNFILLED__` |
| survivor-conditional data boundary | `__UNFILLED__` |
| expected contract/cohort identity | `__UNFILLED__` |
| included/excluded/no-data/error/HOLD row policy | `__UNFILLED__` |
| scheduler/source failure policy | `__UNFILLED__` |
| formation population predicate | `__UNFILLED__` |
| context-request equivalence/exception policy | `__UNFILLED__` |
| typed missingness taxonomy | `__UNFILLED__` |
| feature validity predicate | `__UNFILLED__` |
| proposal eligibility predicate | `__UNFILLED__` |
| fill/no-fill/lapse/invalidation transition | `__UNFILLED__` |
| row and cohort deduplication identity | `__UNFILLED__` |
| full-denominator reconciliation equation | `__UNFILLED__` |

Event-only rows cannot define a training denominator. Watchlist/admission is
metadata, not a silent population filter. Gate position may not determine which
features are observed. The exact `raw_contract_population` definition cannot be
frozen before the P3/U15 decision.

## 4. Causal clocks, evidence availability, and signal SLA

### 4.1 Clock dictionary

For each clock specify source, event semantics, timezone/unit, serialization,
and monotonic/ordering validation.

| Clock | Definition/source | Required before |
|---|---|---|
| `cohort_market_cutoff_ts` | `__UNFILLED__` | P4 |
| `decision_as_of_ts` | `__UNFILLED__` | P4 |
| per-source `expected_closed_boundary_ts` | `__UNFILLED__` | P4 |
| source `request_started_at` | `__UNFILLED__` | P4 |
| source `received_at` | `__UNFILLED__` | P4 |
| source `source_as_of` | `__UNFILLED__` | P4 |
| `sla_reference_attempt_id` | `__UNFILLED__` | P4 |
| SLA reference bar open/close | `__UNFILLED__` | P4 |
| `sla_reference_observed_at` | `__UNFILLED__` | P4 |
| `attempt_deadline_ts` | `__UNFILLED__` | P4 |
| pre-alert recheck clocks | `__UNFILLED__` | P4 |
| `decision_completed_ts` | `__UNFILLED__` | P4 |
| `actionable_ts` | `__UNFILLED__` | P4 |
| local durable publication receipt | `__UNFILLED__` | P5/P10 |
| provider request/acceptance receipt | `__UNFILLED__` | P11 |
| `actionable_channel_id` | `__UNFILLED__` | P5/P11 |
| `actionable_delivery_at` | `__UNFILLED__` | P5/P11 |
| `operator_ack_at` | `__UNFILLED__` | P11 |
| `entry_valid_until_ts` | `__UNFILLED__` | P6 |
| `research_entry_eligible_ts` | `__UNFILLED__` | P6 |
| manual entry/exit clocks | `__UNFILLED__` | P6/P11 |
| typed `label_end_ts` clocks | `__UNFILLED__` | P6/P7 |

### 4.2 Ordering invariants and channel policy

| Field | Required value |
|---|---|
| exact clock inequalities/equalities | `__UNFILLED__` |
| causal cutoff rule | `__UNFILLED__` |
| proposal-linked latest-attempt rule | `__UNFILLED__` |
| higher-high/equal-tick/reset behavior | `__UNFILLED__` |
| pre-alert recheck veto-only semantics | `__UNFILLED__` |
| primary actionable channel | `__UNFILLED__` |
| per-channel receipt definition | `__UNFILLED__` |
| retry/multi-channel semantics | `__UNFILLED__` |
| local-vs-provider-vs-ACK separation | `__UNFILLED__` |
| SLA eligibility predicate | `__UNFILLED__` |
| late/ineligible typed outcomes | `__UNFILLED__` |

`actionable_delivery_at` must be the durable receipt of the frozen actionable
channel. It may not be inferred from decision completion or proposal creation.
The filled contract must preserve every causal-clock invariant in the parent
master plan.

### 4.3 SLA estimands and denominators

| Metric | Formula/denominator | Bound/gate | Interval method |
|---|---|---|---|
| market-peak latency upper bound | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| publication lag | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| operational latency | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| confirmed alerts late rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| SLA-ineligible attempt rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| missed/incomplete fast-cycle rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| universe coverage | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| local/provider p50/p95/p99 | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

## 5. Entry, fill, proposal expiry, and horizon

### 5.1 Deterministic proposal and entry

| Field | Required value | Owner/decision |
|---|---|---|
| proposal side and eligibility | `__UNFILLED__` | hypothesis contract |
| unsigned/unsized geometry inputs | `__UNFILLED__` | proposal contract |
| entry band construction | `__UNFILLED__` | proposal contract |
| proposal validity duration | `__UNFILLED__` | proposal contract |
| invalidation predicates | `__UNFILLED__` | proposal contract |
| primary manual-entry latency | `__UNFILLED__` | U10 |
| latency sensitivity set | `__UNFILLED__` | U10 |
| research fill price rule | `__UNFILLED__` | fill contract |
| same-bar ambiguity rule | `__UNFILLED__` | fill contract |
| no-fill decision rule/time | `__UNFILLED__` | outcome contract |
| lapse rule/time | `__UNFILLED__` | outcome contract |
| invalidated-before-entry rule | `__UNFILLED__` | outcome contract |
| point-in-time instrument revalidation | `__UNFILLED__` | instrument contract |
| concurrency-one arbitration | `__UNFILLED__` | replay contract |
| future no-fill runner-up policy | `__UNFILLED__` | replay contract |

Research entry must use the first causally reachable price at/after its frozen
eligibility time and within the valid band/window. Choosing the best price in a
future window, replacing a future no-fill leader, or backdating a manual fill is
forbidden.

### 5.2 Terminal horizon and label completeness

| Field | Required value | Owner/decision |
|---|---|---|
| primary executable max holding | `__UNFILLED__` | U1 |
| terminal `HORIZON_EXIT` fill rule | `__UNFILLED__` | U1/outcome contract |
| primary label interval | `__UNFILLED__` | split contract |
| shadow observation horizon | `__UNFILLED__` | hypothesis contract |
| shadow diagnostic/target status | `__UNFILLED__` | split contract |
| TP/STOP simultaneous-touch rule | `__UNFILLED__` | outcome contract |
| peak breach interval semantics | `__UNFILLED__` | U7/outcome contract |
| data-gap handling | `__UNFILLED__` | outcome contract |
| delisting handling | `__UNFILLED__` | outcome contract |
| right-censoring handling | `__UNFILLED__` | outcome contract |
| busy-slot release rule | `__UNFILLED__` | replay contract |

Every supervised/selection row needs a typed, causal `label_end_ts`. An observed
timeout without an executable exit cannot release the portfolio slot or become
a terminal PnL label.

### 5.3 Outcome taxonomy

For every state specify predicate, timestamp, required forward evidence, PnL
treatment, label availability, and slot effect.

| Outcome/status family | Frozen definition | Evidence fields |
|---|---|---|
| `TP_FIRST` | `__UNFILLED__` | `__UNFILLED__` |
| `STOP_FIRST` | `__UNFILLED__` | `__UNFILLED__` |
| `HORIZON_EXIT` | `__UNFILLED__` | `__UNFILLED__` |
| `NO_FILL` | `__UNFILLED__` | `__UNFILLED__` |
| `PROPOSAL_LAPSED` | `__UNFILLED__` | `__UNFILLED__` |
| `INVALIDATED_BEFORE_ENTRY` | `__UNFILLED__` | `__UNFILLED__` |
| `RIGHT_CENSORED_DATA_END` | `__UNFILLED__` | `__UNFILLED__` |
| `LABEL_UNAVAILABLE_DATA_GAP` | `__UNFILLED__` | `__UNFILLED__` |
| `LABEL_UNAVAILABLE_DELISTING` | `__UNFILLED__` | `__UNFILLED__` |
| `STATE_RECOVERY_FAILED` | `__UNFILLED__` | `__UNFILLED__` |

## 6. Costs, Cross sizing, and risk

### 6.1 Cost contract

| Component | Source/estimator | Value/distribution | Timing | Stress/sensitivity |
|---|---|---|---|---|
| entry fee | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| exit fee | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| fee schedule/account receipt | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| half-spread | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| normal slippage | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| stop-gap overshoot | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| funding | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| rounding/contract conversion | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| market-impact limitation | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Actual operator/account fees are a separate required receipt; an API fee
announcement or generic schedule may not silently substitute for them.

### 6.2 Sizing and account-risk contract

| Field | Required value | Required before |
|---|---|---|
| equity semantics per research portfolio | `__UNFILLED__` | P6 |
| operator-attested equity semantics | `__UNFILLED__` | P11/P12B |
| private reconciled equity semantics | `__UNFILLED__` | P13 |
| research absolute per-trade cap | `__UNFILLED__` | P6/U11 |
| research equity-fraction cap | `__UNFILLED__` | P6/U11 |
| effective account-leverage cap | `__UNFILLED__` | P6 |
| preferred structural-stop bound | `__UNFILLED__` | P6 |
| absolute adverse-trigger stress bound | `__UNFILLED__` | P6 |
| release-stage caps | `__UNFILLED__` | U11/P11/P15 |
| entry-price-band worst case | `__UNFILLED__` | P6 |
| quantity/notional rounding | `__UNFILLED__` | P6 |
| point-in-time risk-tier constraints | `__UNFILLED__` | P6 |
| selected venue leverage semantics | `__UNFILLED__` | P6 |
| liquidation estimator/version | `__UNFILLED__` | U13/P12B |
| stop-to-liquidation buffer | `__UNFILLED__` | U13/P12B |
| operator attestation max age | `__UNFILLED__` | U13/P12B |
| unknown-state abstention rule | `__UNFILLED__` | P6/P12B |

The maximum rounded-down size must satisfy every frozen loss, leverage,
instrument, tier, and stress constraint after costs and rounding. Selected
exchange leverage may reduce margin requirement but may not expand allowed
notional or account loss. Any unknown account/instrument/liquidation state must
produce non-actionable evidence or `ABSTAIN`.

## 7. Data, split, purge, and holdout construction

### 7.1 Immutable data bindings

| Artifact | Manifest/hash | Time range | Population/universe receipt | Allowed role |
|---|---|---|---|---|
| raw payload pages | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| normalized Min1 partitions | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| derived MTF partitions | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| instrument/funding/OI data | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| point-in-time universe snapshots | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| feature/proposal/outcome dataset | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Discovery-only or survivor-conditional data must be typed as such. It cannot be
promoted to final-holdout or admission evidence by changing a filename.

### 7.2 Chronological split manifest

| Field | Required value |
|---|---|
| train interval | `__UNFILLED__` |
| exploratory development interval | `__UNFILLED__` |
| walk-forward folds | `__UNFILLED__` |
| candidate-selection interval | `__UNFILLED__` |
| final holdout interval/storage boundary | `__UNFILLED__` |
| prospective interval rule | `__UNFILLED__` |
| common calendar | `__UNFILLED__` |
| cycle/minute cohort assignment | `__UNFILLED__` |
| split construction code/version/hash | `__UNFILLED__` |
| split manifest hash | `__UNFILLED__` |
| pre-opening access-control receipt | `__UNFILLED__` |

### 7.3 Purge, embargo, and dependence

| Field | Required value |
|---|---|
| `event_span` definition | `__UNFILLED__` |
| typed `label_end_ts` mapping | `__UNFILLED__` |
| overlap purge algorithm | `__UNFILLED__` |
| primary embargo duration/formula | `__UNFILLED__` |
| shadow-horizon embargo rule | `__UNFILLED__` |
| same-cohort fold rule | `__UNFILLED__` |
| time-block definition | `__UNFILLED__` |
| symbol/cycle clustering | `__UNFILLED__` |
| common-shock dependence policy | `__UNFILLED__` |
| leakage audit/tests | `__UNFILLED__` |

No sample whose event span overlaps validation/test may remain in train. Embargo
must cover entry validity plus the maximum holding/label horizon used in target,
tuning, or admission.

## 8. Comparator and baseline ladder

Every comparator uses the same causal universe, common calendar, entry latency,
fill/expiry rules, proposal geometry, costs, sizing, concurrency-one arbitration,
and no-fill handling as the candidate unless the preregistration explicitly
defines the single intended difference.

| Baseline | Frozen implementation/version | Seeds/matching | Expected invariant | Receipt |
|---|---|---|---|---|
| no-trade economic zero | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| frozen v2/Min60 historical control | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| v2 candidates under common execution | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| simple preregistered Min1 heuristic | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| always-short causal pump-eligible | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `RandomRanking` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `RandomTiming` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| constant prior | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| regularized logistic regression | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Required random-baseline declarations:

| Field | Required value |
|---|---|
| seed list and generator/version | `__UNFILLED__` |
| RandomRanking causal eligible set | `__UNFILLED__` |
| RandomTiming eligible-minute population | `__UNFILLED__` |
| symbol/liquidity/time/regime matching | `__UNFILLED__` |
| calipers, replacement, and failure rules | `__UNFILLED__` |
| opportunity-decision count preservation | `__UNFILLED__` |
| no resample-until-fill proof | `__UNFILLED__` |

## 9. Candidate/model policy

| Field | Required value |
|---|---|
| deterministic rule candidate(s) | `__UNFILLED__` |
| admissible feature roles for direction head | `__UNFILLED__` |
| admissible feature roles for EV head | `__UNFILLED__` |
| model families allowed in trial budget | `__UNFILLED__` |
| hyperparameter search space/budget | `__UNFILLED__` |
| calibration method | `__UNFILLED__` |
| missingness handling | `__UNFILLED__` |
| out-of-fold prediction protocol | `__UNFILLED__` |
| rank/filter/abstain semantics | `__UNFILLED__` |
| artifact serialization + dependency lock | `__UNFILLED__` |
| deterministic inference budget | `__UNFILLED__` |
| drift/rollback policy | `__UNFILLED__` |

ML may rank, estimate outcome probabilities/EV, or abstain only inside the frozen
deterministic proposal/risk envelope. It may not select side, leverage, quantity,
stop, TP, or order action. Public-text LLM output, if ever evaluated separately,
must be timestamped structured context and cannot enter execution authority.

Model fitting is forbidden until P7 is accepted and this preregistration,
population/split manifests, trial budget, gates, and receipts are frozen.

## 10. Primary and secondary economic metrics

### 10.1 Primary profitability gates

| Gate | Estimator | Comparator | Minimum effect | CI/alpha | Pass rule |
|---|---|---|---|---|---|
| absolute economic edge | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| incremental economic edge | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| learned-policy ranking gate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

The primary unit must be a common-calendar portfolio estimator including days
with no position. Passing incremental but failing absolute edge is not edge.

### 10.2 Secondary metrics

| Metric | Exact definition | Interval/aggregation | Role |
|---|---|---|---|
| ending equity | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| cumulative net PnL | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| mean daily net return | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| trade-level EV | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| exposure time | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| maximum drawdown | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| calibration/discrimination | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| MAE/MFE/time-to-event | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| turnover/opportunity rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Secondary metrics cannot rescue a failed primary gate.

## 11. Safety gates

Every cap, confidence bound, direction, aggregation unit, and familywise policy
must be frozen before candidate-model development performance is inspected.

| Safety metric | Exact estimator | Upper bound | CI/alpha | Pass rule |
|---|---|---|---|---|
| stop-first rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| peak-breach-before-exit rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| adverse-move exceedance probability | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| adverse-move expected shortfall | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| account-loss expected shortfall | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| portfolio maximum drawdown | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| SLA miss rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| incomplete/missed-cycle rate | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Required safety sensitivities/stresses:

| Stress | Frozen grid/distribution | Pass/fail use |
|---|---|---|
| fees | `__UNFILLED__` | `__UNFILLED__` |
| manual-entry latency | `__UNFILLED__` | `__UNFILLED__` |
| spread/slippage | `__UNFILLED__` | `__UNFILLED__` |
| stop-gap overshoot | `__UNFILLED__` | `__UNFILLED__` |
| funding | `__UNFILLED__` | `__UNFILLED__` |
| liquidity/contract-size strata | `__UNFILLED__` | `__UNFILLED__` |
| market-regime/calendar blocks | `__UNFILLED__` | `__UNFILLED__` |

Both profitability gates and every safety gate must pass. A profitable but unsafe
candidate receives the unsafe verdict; no weighted composite may hide a safety
failure.

## 12. Power and uncertainty design

| Field | Required value | Evidence source allowed before freeze |
|---|---|---|
| one-/two-sided test convention | `__UNFILLED__` | design only |
| alpha and interval method | `__UNFILLED__` | design only |
| target power | `__UNFILLED__` | design only |
| absolute minimum effect | `__UNFILLED__` | economic/user criterion |
| incremental minimum effect | `__UNFILLED__` | economic/user criterion |
| safety-cap margins | `__UNFILLED__` | economic/user criterion |
| effective independent unit | `__UNFILLED__` | mechanics/design data |
| dependence/block-length policy | `__UNFILLED__` | mechanics/design data |
| symbol/cycle cluster policy | `__UNFILLED__` | mechanics/design data |
| sample-size/power method | `__UNFILLED__` | mechanics/design data |
| minimum independent blocks | `__UNFILLED__` | design only |
| minimum eligible opportunities | `__UNFILLED__` | design only |
| minimum entered trades | `__UNFILLED__` | design only |
| minimum calendar duration | `__UNFILLED__` | design only |
| stopping rule | `__UNFILLED__` | design only |
| matched-random replication count | `__UNFILLED__` | design only |
| bootstrap/permutation seed manifest | `__UNFILLED__` | design only |
| power calculation code/version/receipt | `__UNFILLED__` | design only |

Raw trade count is not independent sample size. Power design may use mechanics
or blinded design data but may not use candidate-model development performance
to weaken minimum effects, safety caps, alpha, or stopping rules. Insufficient
power yields `INCONCLUSIVE`, never a positive or negative edge claim.

## 13. Multiple testing and negative controls

### 13.1 Trial family and budget

| Trial dimension | Allowed values/count | Selection level | Correction/budget |
|---|---|---|---|
| hypothesis versions | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| feature-contract variants | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| rule thresholds | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| TP geometry choices | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| entry latency choices | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| cost/stress choices | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| model families | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| hyperparameter trials | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| probability/EV thresholds | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| random matching variants | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

| Global declaration | Required value |
|---|---|
| family definition | `__UNFILLED__` |
| nested-selection protocol | `__UNFILLED__` |
| correction method | `__UNFILLED__` |
| total trial budget | `__UNFILLED__` |
| append-only trial ledger path/hash | `__UNFILLED__` |
| failed/abandoned trial accounting | `__UNFILLED__` |
| single champion freeze rule | `__UNFILLED__` |
| unplanned-analysis/new-version rule | `__UNFILLED__` |

### 13.2 Negative controls

| Control | Frozen implementation | Expected result | Failure action |
|---|---|---|---|
| label permutation | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| timestamp shift | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| random features | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| future-data sentinel | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| no-fill runner-up sentinel | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| worker/order/restart invariance | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

## 14. Verdict taxonomy and decision algorithm

Persisted verdict enum:

```text
DATA_INVALID
INCONCLUSIVE
NO_EDGE_FOR_FROZEN_HYPOTHESIS
EDGE_CANDIDATE_OFFLINE_UNSAFE
EDGE_CANDIDATE_OFFLINE
EDGE_CONFIRMED_PROSPECTIVE
SIGNALS_OPERATION_VALIDATED
```

| Verdict | Exact Boolean/ordered predicate | Required receipts | Permitted next action |
|---|---|---|---|
| `DATA_INVALID` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `INCONCLUSIVE` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `NO_EDGE_FOR_FROZEN_HYPOTHESIS` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `EDGE_CANDIDATE_OFFLINE_UNSAFE` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `EDGE_CANDIDATE_OFFLINE` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `EDGE_CONFIRMED_PROSPECTIVE` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| `SIGNALS_OPERATION_VALIDATED` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

The filled decision algorithm must be deterministic and ordered so that data
invalidity, inadequate power, profitability failure, and safety failure cannot
be overwritten by a favorable secondary metric. `EDGE_CONFIRMED_PROSPECTIVE`
requires a separate frozen future protocol; an offline holdout cannot produce it.

## 15. One-time final holdout opening

### 15.1 Pre-opening checklist

Every box is intentionally empty. Any empty box blocks opening.

- [ ] hypothesis/version and parent hashes frozen
- [ ] data/population/split manifests frozen and reconciled
- [ ] no final-holdout data accessed by candidate-selection code or operator
- [ ] feature/strategy/proposal/cost/risk/outcome contracts frozen
- [ ] baseline implementations, seeds, and matching frozen
- [ ] primary/secondary metrics and every safety gate frozen
- [ ] alpha, power, dependence, minimum effects, and stopping rule frozen
- [ ] multiple-testing budget and complete trial ledger frozen
- [ ] one champion artifact and deterministic environment frozen
- [ ] negative controls and leakage/restart/invariance tests pass
- [ ] all required U1–U15 decisions for this stage signed
- [ ] independent reviewer approval recorded
- [ ] detached pre-opening checkpoint receipt recorded
- [ ] explicit one-time opening authority recorded

### 15.2 Opening receipt

| Field | Required value |
|---|---|
| holdout ID and immutable manifest hash | `__UNFILLED__` |
| hypothesis/champion/preregistration hashes | `__UNFILLED__` |
| authorized opener | `__UNFILLED__` |
| independent witness/reviewer | `__UNFILLED__` |
| explicit authority receipt | `__UNFILLED__` |
| opening timestamp UTC | `__UNFILLED__` |
| exact evaluation command/code/environment hash | `__UNFILLED__` |
| stdout/stderr/result artifact hashes | `__UNFILLED__` |
| pre/post storage/checkpoint hashes | `__UNFILLED__` |
| access/audit log receipt | `__UNFILLED__` |
| automatic verdict + decision trace | `__UNFILLED__` |
| exception/incident record | `__UNFILLED__` |

The holdout is opened once for one frozen champion. A code, feature, threshold,
TP, latency, cost, matching, model, cap, or metric change after opening creates
a new hypothesis version and requires a new future holdout; the opened result is
retained, not overwritten.

## 16. Prospective research-shadow protocol

This section is filled only after an admissible offline verdict. Completing it
does not authorize Telegram, private API, or real-money trading.

| Field | Required value |
|---|---|
| eligible offline verdict/receipt | `__UNFILLED__` |
| frozen champion artifact/hash | `__UNFILLED__` |
| prospective protocol/version/hash | `__UNFILLED__` |
| prospective start rule/time | `__UNFILLED__` |
| prospective end/stopping rule | `__UNFILLED__` |
| minimum calendar duration | `__UNFILLED__` |
| minimum independent blocks/opportunities | `__UNFILLED__` |
| immutable point-in-time universe policy | `__UNFILLED__` |
| local durable actionable channel | `__UNFILLED__` |
| deterministic latency/fill policy | `__UNFILLED__` |
| frozen costs/sizing/risk policy | `__UNFILLED__` |
| scheduler/coverage/SLA gates | `__UNFILLED__` |
| drift and missing-data gates | `__UNFILLED__` |
| incident/pause/restart policy | `__UNFILLED__` |
| no-tuning enforcement/audit | `__UNFILLED__` |
| blinded monitoring fields | `__UNFILLED__` |
| unblinding/evaluation authority | `__UNFILLED__` |
| matched baseline replay protocol | `__UNFILLED__` |
| prospective profitability gates | `__UNFILLED__` |
| prospective safety gates | `__UNFILLED__` |
| promotion/rollback verdict algorithm | `__UNFILLED__` |

During the prospective window, features, thresholds, TP, latency, costs,
matching, model, caps, and gates cannot be tuned. Operational incidents remain
in the denominator with typed status. A failure pauses or invalidates according
to the frozen rule; it is not silently removed. Provider and operator workflows
remain separate later gates.

## 17. U1–U15 decision and sign-off register

No row is signed by this template. A row is complete only with a selected value,
scope/version binding, evidence receipt where required, named approver, UTC
timestamp, and explicit status. The parent master plan defines each deadline.

| ID | Decision requiring sign-off | Selected value/policy | Scope/version | Evidence receipt | Approver + UTC | Status |
|---|---|---|---|---|---|---|
| U1 | executable maximum holding and terminal horizon | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U2 | pre-alert recheck source and semantics | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U3 | safety upper bounds and freeze timing | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U4 | actual manual/operator fee schedule | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U5 | explicit public-pilot network permission | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U6 | dedicated empty MEXC subaccount | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U7 | peak breach as safety evidence or exit | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U8 | TP candidate set and frozen champion rule | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U9 | final automation/release scope | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U10 | primary manual-entry latency and sensitivity set | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U11 | versioned research/operational/live risk caps | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U12 | runtime RPO/RTO/retention | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U13 | attestation age and liquidation estimator/buffer | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U14 | canary promotion evidence | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |
| U15 | fallback if full-universe Min1 is infeasible | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` | `__UNFILLED__` |

Explicit reminder: U5 remains ungranted until a separate user authorization is
recorded. This file is not that authorization. U6/U13/U14 are later operational
or live-stage decisions and do not permit credential or order access now.

## 18. Final completeness and phase-release checklist

Every box is intentionally empty. The checker must reject the document while a
required box or sentinel remains.

- [ ] no `__UNFILLED__` remains in fields required for the requested phase
- [ ] all referenced versions/hashes resolve to immutable artifacts
- [ ] parent-plan invariants are represented without weakening
- [ ] exact population denominators reconcile
- [ ] causal clocks and actionable-channel receipt invariants validate
- [ ] entry/fill/horizon/outcome semantics are executable and causal
- [ ] costs, Cross sizing, rounding, and stress losses are exact
- [ ] split, purge, embargo, and point-in-time universe are auditable
- [ ] comparator ladder shares one execution/calendar contract
- [ ] primary profitability and every safety gate are frozen
- [ ] power and multiple-testing budgets are frozen before result access
- [ ] negative controls and leakage tests are specified
- [ ] one-time holdout opening procedure is independently approved
- [ ] prospective no-tuning protocol is frozen before its start
- [ ] all U1–U15 decisions required for the requested phase are signed
- [ ] final canonical hash and detached checkpoint receipt exist

Phase-release rule:

```text
template created                != preregistration frozen
preregistration frozen          != public-pilot permission
offline holdout passed          != prospective edge
prospective edge confirmed      != signals operations validated
signals operations validated    != private/live execution authority
```

Until the relevant fields, receipts, reviews, and user authorities are complete,
the only permitted state of this document is `TEMPLATE_ONLY / BLOCKED`.
