# AI collaboration handoff

Updated: 2026-08-16, America/Chicago

> [!IMPORTANT]
> Fresh-session entrypoint: read `CLAUDE.md`, then treat
> `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md` as the authoritative final-product
> roadmap. Its independent Claude closure review returned
> `APPROVE_AS_AUTHORITATIVE` on 2026-08-15 with no remaining P0/P1. The matching
> v3 ADR and preregistration skeleton named by `CLAUDE.md` are mandatory next
> reads. `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` remains a frozen
> historical/executable-v2 audit source, not the current product roadmap. The
> approval-base HEAD is `ad30b02`; the authoritative S1 roadmap publication is
> `2a14299`. The published AI
> foundation anchor is `f0b43d6`; the StrategySpec/journal-v5
> foundation tip is `2d0efcb`, the frozen-behavior tip is `258c35f`, the
> versioned StrategySpec evidence tip is `1971b77`, the finalized typed
> lifecycle tip is `9ef6b4f`, and the frozen-v2 journal/runtime tip is
> `bb1ca13`. The completed S3 aggregation tip is `0ff1b3a`; the completed S2
> strict-history tip is `36e1446`. The bounded transport tip is `ba8ea00` and
> the restart-safe strict-history-v2 tip is `f8a6b5b`; their roadmap receipt is
> `17b47c7`. The current bounded offline P2 QA-pilot run-contract tip is
> `5595679`.
> The later `ad30b02` research dataset-builder descendant is not v3/model
> evidence. The
> `98217df`/`340 passed`
> checkpoint immediately below is a
> preserved earlier causal-scanner snapshot, not the latest foundation state.
> Latest code validation: `925 passed, 5 skipped, 2 known collection warnings`.
> The pilot-contract slice also has focused `20 passed` and all-`tests/v3`
> `194 passed, 1 skipped` receipts. Current writer/default path: journal schema v6 at
> `data/runtime/mexc_population_decisions_v6.jsonl`. Schema v5 remains frozen,
> fixture-backed and read-only; it is not the current append target.

## Earlier causal-scanner checkpoint — 2026-08-03

This section supersedes the historical scanner-state statements below. The
earlier audits are retained because their research findings and the reasons for
the fixes remain useful, but forming-bar evaluation, observation backfill,
cross-thread strategy-state contamination, duplicate population cycles, false
Telegram delivery confirmation, and missing population-journal wiring have now
been addressed by the committed causal scanner series.

### Repository and validation state

- MEXC remains the explicitly selected target exchange and this worktree remains
  the implementation base.
- Local branch: `claude/codex-project-review-04581e`.
- Tracking branch: `origin/claude/codex-project-review-04581e`.
- Local HEAD and the remote tracking ref both point to `98217df`; the branch is
  neither ahead nor behind. This handoff update is the only expected
  uncommitted follow-up change.
- Final causal scanner series, inclusive from `e3dcd45` through `e945206`:

```text
e3dcd45 feat(observation): add deterministic population decision journal
bdcdf51 fix(market-data): enforce explicit closed-bar cutoffs
42d972b fix(strategy): isolate mutable signal state during parallel scans
58de7bb fix(observation): make population cycles complete and deduplicated
2a04ebd fix(market-data): preserve the empty closed-frame contract
5c08afe fix(alerts): report confirmed Telegram delivery
e945206 feat(scanner): journal causal decisions from closed MEXC bars
98217df feat(backtest): define causal single-position contract
```

- Validation on the final committed source tree:

```text
full pytest: 340 passed, 4 skipped, 2 warnings
```

### Final causal scanner contract

One invocation of `scan_once()` now has one point-in-time universe and one
causal clock:

1. Refresh and freeze the ordered `UniverseSnapshot`.
2. Capture one `scan_observed_at` and derive one fixed-interval
   `candle_cutoff_ts` with `closed_boundary_ts()`.
3. Fetch both the BTC benchmark and every universe symbol through
   `fetch_closed_frame(..., as_of=candle_cutoff_ts)`. The same cutoff is used by
   every worker. Empty MEXC responses preserve a timezone-aware empty-frame
   contract and become `no_data`, not malformed bars.
4. Validate the frame metadata against its actual last closed bar. The strategy
   receives the last closed `close` as `mark_price`, never the later live ticker,
   and receives the same cutoff through `StrategyContext.candle_cutoff_ts` so
   higher-timeframe reads are bounded by the decision clock.
5. Run the shared layered strategy with its mutable diagnostics, confirmation
   state, and volatility context isolated under the committed thread-safety
   contract. Workers return immutable typed outcomes and do not mutate shared
   skip counters.
6. Capture `decision_ts` only after strategy generation returns. Aggregate on
   the main thread in the original universe order.

The population journal records exactly one `PopulationDecision` for every
ordered universe symbol, including entries, holds and failures. Supported
statuses are `evaluated`, `no_data`, `short_history`,
`invalid_bar_contract`, `data_error`, `data_quality_error`, and
`strategy_error`. Real base-bar open/close timestamps are mandatory whenever a
bar exists and remain `null` when no valid bar exists. Exceptions are persisted
only as `safe_error_code`; exception messages and tracebacks are excluded.

`cycle_id`, `input_hash`, and `snapshot_id` are canonical SHA-256 identifiers.
Equivalent timeframe aliases hash through their fixed duration, wall-clock
decision timestamps do not change causal IDs, and the legacy timestamp-based
signal ID is excluded from causal metadata. Every row carries `cycle_ordinal`
and `cycle_size`; `append_cycle()` rejects incomplete, unordered or duplicate
symbol batches. A repeated final `cycle_id` is not appended and its repeated
signal is suppressed.

Observation tracking uses `snapshot_id` as the stable signal ID,
`signal_ts=decision_ts`, and `signal_bar_ts` equal to the last closed bar's open
timestamp. Previous observations are updated only from available closed frames;
new LONG entries are not passed to the SHORT tracker. A new SHORT observation is
initially recorded as undelivered. It is marked delivered only after an alert
channel returns confirmed success; a configured channel alone is not delivery.

The public `scan_once()` return contract remains a list of entry tuples. The
runtime population journal defaults to
`data/runtime/mexc_population_decisions.jsonl`; it can be redirected with
`--population-journal` or explicitly disabled with
`--disable-population-journal`.

### Executable single-position research contract

Commit `98217df` adds `backtesting/single_position.py`. It is deliberately
separate from the historical DCA replay and is the only contract that future
model labels may target. It is offline/research-only and has no order-submission
path.

The v1 contract is explicit and fail-closed:

- SHORT only, one market entry at the first bar open exactly at `decision_ts`,
  one absolute stop, one absolute take-profit, no DCA and global concurrency
  exactly one;
- a complete gap-free closed-bar horizon is mandatory; stale setups already
  beyond stop or target at entry are recorded as unfilled and receive no
  artificial win;
- stop wins a same-bar stop/target ambiguity, a stop gap exits at the worse open,
  and a target gap receives no optimistic price improvement;
- quantity is bounded simultaneously by equity risk budget, maximum quote
  notional, maximum leverage, quantity step, minimum quantity and minimum
  notional;
- entry/exit fees, half-spread and directional slippage are explicit inputs;
  timestamped funding is explicit, with positive rates paying a SHORT;
- intrabar exits receive only funding known by that bar's open, because OHLC
  cannot prove that a later payment occurred before the exit;
- the result reports fills, reason, timestamps, size, gross PnL, fees, funding,
  net PnL, return on notional and return on risk;
- chronological portfolio selection ranks simultaneous candidates only by their
  causal score and prevents overlap until the selected position exits.

The focused contract suite has 11 tests and is included in the 340-test full
MEXC result above. This defines mechanics; it does not demonstrate edge.

### Safety and credentials boundary

- The MEXC scanner is signals-only and uses public market data. It has no MEXC
  private adapter and no code path that can place, amend or cancel an order.
- Operational mode remains observation/paper-only. The scanner and trading bot
  are stopped; no live capital is authorised and the no-edge finding remains in
  force.
- Credential rotation is deliberately deferred to the user. Because `.env` was
  historically tracked, existing Bybit keys/secrets, Telegram bot token and any
  proxy credentials must be treated as compromised until rotated.
- Deferred rotation does not block local tests, offline research or collection
  from credential-free public MEXC endpoints. It does block all private API,
  testnet/live execution and outbound Telegram use. Do not load or reuse the
  historical credentials merely because the delivery path now reports success
  correctly.

### Required next order

1. **Wire versioned labels to the frozen contract.** Build every future outcome
   from `SinglePositionContract` v1, persist the complete contract/config
   fingerprint with each dataset, and compare its behaviour with independent
   hand-computed fixtures. Do not reuse the historical DCA labels.
2. **Collect the runtime population.** Run the signals-only scanner in controlled
   paper observation, preserve complete point-in-time universe cycles and all
   HOLD/error rows, and monitor coverage and data-quality statuses. Do not build
   the next dataset only from fired events.
3. **Run purged chronological validation.** Use decision-time cohort boundaries,
   a global purge/embargo at least as long as the 48-hour label horizon,
   point-in-time universe inputs, per-timeframe warm-up/cadence validation,
   symbol-clustered confidence intervals and a paired random-entry baseline.
4. **Replace the legacy trainer before training.** The current `ai/train.py`
   still targets `target_win/target_horizon`, selects XGBoost before LightGBM in
   `auto` mode, uses a simple 80/20 split and fits its calibrator on that test
   block. It is not the new evaluation path and must remain disconnected.
5. **Establish the first ML baseline.** Fit a small CPU LightGBM model on the
   causal tabular runtime-population features to estimate TP-first, SL-first and
   timeout/net-EV outcomes. Calibrate on a later past-only validation slice and
   compare `rules only`, random entry and `rules + LightGBM` on untouched future
   folds after all costs. Keep it shadow/paper-only and accept another no-edge
   result without threshold or model shopping.

Private MEXC execution and any live deployment remain a separate later project,
conditional on repeated, reproducible future-fold edge under the frozen
single-position contract.

## Independent Codex review

Scope: `claude/codex-project-review-04581e` at
`68e0ff77373db2b48c19e3dcf09f1cbb7d569e47`.

- User decision: MEXC is the target exchange and this Claude branch is the
  implementation base; Codex is the independent reviewer.
- No merge, strategy change, threshold change, or source-code change was made.
- The bot was not started. It remains stopped.
- The complete review, including the retained history of the other worktree, is
  also available at
  `C:\Users\vlasy\PycharmProjects\koteika_Ultra\docs\AI_HANDOFF.md`.

## A. Look-ahead damage

MEXC kline time is the bar's open time. The 1h signal is knowable only at its
close, so the honest forward window is
`[event.ts + 3600, event.ts + 3600 + 48h]`. The current worktree contains this
fix in `ai/pump_dataset.py:285-304`; the existing
`data/processed/pump_dataset_v2.csv` was built before the fix.

Requested paired check:

- 25 deterministically stratified symbols.
- 1,378 complete paired 5m event windows.
- Old-window reconstruction matched every stored `mae_pct`, `mfe_pct`,
  `dca_resolved`, and `n_averages` value.
- Confidence intervals use 50,000 symbol-cluster bootstrap draws.

| Metric | Old from `event.ts` | Corrected `+1h` | Change |
|---|---:|---:|---:|
| Mean `mae_pct` | 15.8689% | 15.9153% | +0.0464 pp |
| Mean `mfe_pct` | 10.5547% | 10.1989% | -0.3558 pp |
| `dca_resolved` | 91.3643% | 86.7925% | **-4.5718 pp** |
| Mean `n_averages` | 0.2409 | 0.4419 | **+83.43%** |

- MAE delta 95% cluster CI: `-0.0425..+0.1106 pp`.
- Resolved-rate delta CI: `-6.5028..-3.1295 pp`.
- Mean-add absolute delta CI: `+0.1257..+0.2630`.
- 65 events changed resolved to unresolved; only 2 reversed.
- 664/1,378 (48.19%) had falsely reached the 3% target inside the already
  elapsed signal hour.
- All 65 lost resolutions and 157/161 increased-add cases came from that
  contaminated hour.

Full 5,042-event sensitivity check:

| Metric | Old | Corrected | Change |
|---|---:|---:|---:|
| Mean `mae_pct` | 17.6074% | 17.6656% | +0.0582 pp |
| Mean `mfe_pct` | 11.9100% | 11.5908% | -0.3192 pp |
| `dca_resolved` | 92.4831% | 88.6355% | **-3.8477 pp** |
| Mean `n_averages` | 0.2376 | 0.4351 | **+83.14%** |

For the reported final filter (240 events), the correction changes resolved
from 100.000% to 98.333%, mean adds from 0.254 to 0.692, and worst code
drawdown from 1.260 to 5.684. A 3%-consistent peak-loss score changes from
`+0.03861` to `-0.01251`.

Verdict: cosmetic for aggregate MAE, but material for recovery, averaging depth,
worst drawdown, and expectancy. Rebuild the CSV before using any old conclusion.

## B. Min1 versus Min60 mismatch

No MEXC Min1 cache existed. The exact check therefore used fresh completed
public MEXC candles on 2026-07-26, without credentials and without starting the
bot. Ten symbols at 33 aligned hourly points gave 330 paired observations.
Production indicator code and the exact `sum(close*volume)` gate formula were
used.

| Comparison | Result |
|---|---:|
| Median Min60/Min1 ATR-percent ratio | **10.34x** |
| ATR ratio p10 / p90 | 6.25x / 18.56x |
| Median Min60/Min1 12-bar volume ratio | **89.18x** |
| Volume ratio p10 / p90 | 26.65x / 471.07x |

| Threshold | Min1 pass | Min60 pass |
|---|---:|---:|
| `atr_pct >= 0.046` | **0/330 (0.00%)** | 59/330 (17.88%) |
| `sum(close*volume, 12) >= 100000` | 76/330 (23.03%) | 198/330 (60.00%) |

Twelve Min1 bars are 12 minutes; twelve Min60 bars are 12 hours. The original
volume calibration used twelve Min5 bars, or one hour. These are different
horizons. In addition, MEXC `vol` is multiplied by price without contract size,
so the value is not guaranteed to be USDT notional across contracts.

Verdict: the Min60 thresholds are not portable to Min1. In this sample the ATR
gate blocks every Min1 observation.

## C. Expectancy audit

There is no committed expectancy function. Claude scratchpad/session commands
used:

```text
score = target * (n_averages + 1)   if dca_resolved
        -dca_peak_drawdown_units    otherwise
```

The dataset labels have `dca_target_pct=0.03`
(`ai/pump_dataset.py:43-49`). Later evaluations substituted `target=0.05`
without replaying `dca_resolved`, adds, or drawdown at 5%. The apparent
improvement is therefore built into the formula.

| Actual target/replay | Actual resolved | Correct replay score | Fixed-3%-label score |
|---|---:|---:|---:|
| 2% | 97.660% | +0.011148 | -0.011212 |
| 3% | 92.483% | -0.000391 | -0.000391 |
| 5% | 78.144% | **-0.031719** | **+0.021251** |

On the final rule's chronological test, the reported 5% score changes from
`+0.059524` to `-0.044530` when events are actually replayed at 5%.

`dca_peak_drawdown_units` is worst floating loss, not a realized exit. For 379
unresolved old-window events, peak loss mean/median was 0.4371/0.0836, versus
terminal 48h MTM loss 0.2190/0.0367. A no-cost horizon close changes all-event
E3 from `-0.000391` to `+0.016002`. Peak-as-realized is pessimistic but
non-executable.

The pro-averaging bias is in the resolved branch:

- every eventual target touch receives `+target * legs` while survived drawdown
  is ignored;
- 59 old-window winners survived more than 0.5 initial-leg drawdown units, 29
  survived more than 1, and the maximum was 11.843;
- extra legs are rewarded without dividing by deployed capital or modeling
  capital occupancy/concurrency;
- stops/liquidation, fees, spread, slippage, funding, margin and cash limits are
  absent;
- code DD understates initial-entry-normalized DD by up to 24% at six adds;
- arithmetic blended entry is valid for equal quantity, not equal USDT legs;
- high-then-low 5m ordering is a real but small bias (about 0.64% of winners).

Verdict: this score is a recovery/path diagnostic at one fixed target, not
trading expectancy. It cannot establish profitability or DCA safety.

## D. Runtime wiring audit

Current path:
`app/main.py:165-177` fetch/build -> `:179-192` strategy ->
`:194-211` risk/execution -> `:247-251` alert.

| Component | Runtime state | Connection point |
|---|---|---|
| MEXC public feed | Not wired; `MarketDataFeed` is created with `api.bybit.com`. | Replace/refactor `app/main.py:279-290`; inject the MEXC client. |
| MEXC private execution | Not implemented; runtime uses `BybitAdapter`. | New adapter before replacing `:286`; it must satisfy `:101`, `:118-126`, `:194-211`, `:290-310`, and startup `:332-337`. |
| Dynamic MEXC universe | Offline only; runtime uses static `BOT_SYMBOLS`. | Construct before `:286-287`; refresh before `:332` and at `:363`; replace `cfg.symbols` at `:333`/`:365` and update WS membership. |
| Sentiment | **Already connected.** | `:319-330`, `:103-106`, `:171-173`, `:187-188`. |
| Funding/OI/long-short | Not connected; two values are hardcoded `None`. | Populate at `:171-177`; extend context at `:180-192`. |
| Observation tracker | Not connected in this branch. | Instantiate `:296-318`; update after `:165-170`; record around `:180-192`/`:247-251`; expire near `:382-385`. |
| Position tracker | Not connected in this branch. | Instantiate/pass similarly; mark after `:179`, record at `:247-251`, close around `:211-214`. |
| Chart/liquidation panel | No runtime call site/photo path. | Build after `:177-192`; send at `:247-251`; expose photo sending. |
| HTF cache | Formally connected at `:313-316`, but `Hour4` goes through the Bybit feed and failure is fail-open. | Correct backend at `:289`; require data-quality status before `:180`. |
| `min_rsi_1h` / `require_confluence` | Declared, never consumed. Fib, divergence, confluence, levels and liquidation features remain offline. | Build closed-bar MTF context after `:177`, before `:180`. |
| BTC relative strength | Connected but fail-open and currently follows Min1. | Fetch/validate calibrated Min60 benchmark at `:108-114`. |
| Cross-sectional volatility | Connected but inherits timeframe mismatch. | Supply calibrated data at `:165-180`. |
| Overhead gate / `max_chase_atr` | Implemented, disabled by defaults, no runtime override. | Explicit config at strategy construction `:311-312`; do not activate from biased labels. |
| Other pump features | Exhaustion, wick, acceleration, idiosyncratic return, EMA50 and consecutive-up remain dataset-only. | Select validated fields in pipeline `:177` and context `:180-192`. |
| ML inference | Pipeline exists, but no inference reaches strategy. | Shadow inference between `:177` and `:180`. |
| DCA, partial TPs, safe leverage | Calculated/reported offline; one runtime intent does not execute that model. | Reporting at `:247-251`; real execution requires a tested state machine across `:194-211`. |

Offline history, dataset building, and labels should remain offline. Runtime
needs a closed-bar MEXC data/universe adapter and a small validated feature
contract, not the research pipeline itself.

## Required next order

1. Rebuild labels after `decision_ts`.
2. Replay every target at the actual target.
3. Define executable PnL, sizing/capital, exit/stop/liquidation and costs.
4. Calibrate and run on identical timeframe, horizon and volume units.
5. Wire/validate MEXC public data and dynamic universe before private execution.
6. Keep the bot stopped; this review gives no basis for live trading.

## Codex review checkpoint — 2026-07-26 01:46 MSK

Reviewed Claude worktree at `8cc31fc` with an uncommitted
`ai/pump_dataset.py` diff.

The current draft correctly:

- makes the forward window half-open at `decision_ts + horizon`;
- anchors `time_to_target_min` to the explicit decision timestamp;
- rejects low-coverage or materially gapped forward histories;
- emits forward-window quality metadata.

Existing targeted tests remain green:

```text
tests/v2/test_pump_dataset_v2.py: 14 passed
```

The draft is not review-complete yet:

- no regression tests cover the exact-horizon exclusion, a missing first bar,
  incomplete/gapped history, or the end-to-end `build_symbol_rows` path;
- `forward_window_quality()` counts rows rather than unique aligned timestamps,
  so duplicate bars can inflate coverage and mask missing bars;
- timestamp alignment/order should be validated explicitly before treating
  coverage as trustworthy;
- the chosen closed-4h feature semantics still need a focused regression test.

Do not commit or rebuild datasets until those cases are covered and reviewed.

## Codex review checkpoint — 2026-07-26 02:08 MSK

The follow-up dataset draft and new tests were reviewed without changing
strategy/runtime code.

Confirmed good:

- production forward selection is now half-open;
- `label_event()` receives the explicit `decision_ts`;
- focused dataset suite passes: `26 passed`;
- the worktree is still only one commit ahead of its tracking ref; no current
  HEAD push/merge was observed and no Python bot process is running.

Blocking findings:

1. `forward_window_quality()` counts rows, not unique slots on the expected
   cadence. A probe with 48 rows representing only 41 unique slots reports
   `coverage=1.0`, `max_gap_bars=2.0` and passes. A complete set shifted off the
   decision grid also reports `coverage=1.0` and passes. Validate finite,
   strictly unique, cadence-aligned timestamps before calculating coverage.
2. The `0.90` default permits roughly 4.8 hours of scattered missing history
   in a 48-hour window. That can still hide MAE/MFE/target extremes; the allowed
   loss of coverage needs evidence or a substantially stricter contract.
3. `test_forming_four_hour_bar_is_excluded_end_to_end` does not assert any
   output feature. It independently filters the fixture and therefore passes
   even if `build_symbol_rows()` consumes the forming bar. Poison the forming
   bar and assert an actual output (including the separately resampled
   `rsi_4h`). Note that `build_features()` currently resamples the last
   available hourly bars into a potentially partial UTC 4h bucket, which is
   inconsistent with the test's declared closed-4h semantics.
4. The newly added replay draft is not stable: at this checkpoint
   `backtesting/replay.py:130` references undefined `_LEVEL_EPS`; focused replay
   tests are `8 failed, 5 passed`. In addition, the within-bar adverse path must
   compare the next stop and next DCA level in price order. Filling every DCA
   level first can move a configured stop that would have been crossed before
   the first add.

Do not use the replay results, rebuild the dataset, or commit this draft until
the above cases are fixed and the full suite is green.

## Codex replay audit — 2026-07-26 02:16 MSK

Correction to the prior checkpoint: Claude subsequently defined `_LEVEL_EPS`
and adjusted the replay tests. Current validation is:

```text
focused dataset + replay: 39 passed
full suite: 243 passed, 4 skipped, 3 warnings
```

Green tests do not resolve the replay semantics:

- Stop and DCA thresholds are still not processed in path order. With entry
  `100`, stop `2%`, first DCA `108`, and bar high `109`, the model adds at 108
  first and exits two legs at `105.923`, returning `-4%` on the initial leg.
  A rising path necessarily crosses the active stop at `102` before the add.
- Stop fills ignore gaps. For a one-leg short at 100 with stop 102 and a bar
  whose entire range is 110–120, the model reports a fill at 102 even though
  that price was unavailable. Either require/open-aware bars and fill at the
  worse of stop/open, or label this as an optimistic lower bound.
- The comment that default MEXC costs were “measured live” has no provenance in
  the repository. Preserve symbol/time/sample statistics or make the values
  explicit scenario inputs rather than factual defaults.
- Funding, liquidation/margin constraints, simultaneous capital occupancy and
  portfolio cash limits remain absent. This module may become a pathwise trade
  diagnostic, but it is not yet executable portfolio expectancy.

The bot remains stopped. No strategy/runtime changes, dataset rebuild, commit,
merge, or push were performed by Codex.

## Codex label-semantics audit — 2026-07-26 02:46 MSK

No Claude code changed during this interval. An additional inconsistency was
reproduced in the existing labels:

- `mae_pct` and `mfe_pct` are calculated over the entire forward horizon before
  the DCA path is replayed.
- The DCA loop stops at the first target resolution, so `n_averages`,
  `dca_resolved` and drawdown only describe exposure until exit.

Example: entry 100, an immediate low at 95 resolves the short with zero adds;
a later, post-exit bar at 200 makes the same row report `mae_pct=1.0`,
`dca_resolved=1`, `n_averages=0`. The `good_mae_*` flags then classify a move
that happened after the modeled trade had already closed.

Choose and name one of two semantics before rebuilding:

1. **Trade-path labels:** MAE/MFE stop at the modeled exit and remain consistent
   with DCA legs/drawdown; or
2. **Fixed-horizon market labels:** retain full-horizon MAE/MFE, but do not
   describe them as required averaging depth or executable trade risk.

Add an immediate-target-then-later-pump regression test for the chosen
contract. Current green tests do not exercise this distinction.

## Codex history-cache audit — 2026-07-26 03:16 MSK

No Claude code changed during this interval. The existing cached MEXC history
was checked read-only:

```text
files scanned: 611
duplicate/off-grid/unsorted files: 0
files with cadence gaps: 24
gap records: 31
missing expected slots: 46,339
```

Many holes are exact MEXC page multiples (2,000/4,000 bars). For example,
`PROMUSDT_Min5.csv` is missing 10,000 expected bars and
`SYNUSDT_Min5.csv` is missing 4,000. `BTWUSDT` has a long aligned gap across
Min5/Min15/Min60/Hour4. Some gaps may be genuine trading suspensions, but the
collector cannot distinguish that from a failed request:

- `_request_public()` returns `None` after retry exhaustion;
- `_fetch_window()` converts both request failure and valid no-data to an empty
  frame;
- `HistoryCollector.fetch_range()` advances to `window_end` when the frame is
  empty and later writes the partial cache.

Impact on the current artifacts:

- `pump_dataset_v3.csv`: 16 rows have a 48h forward window overlapping a known
  Min5 gap. The new forward guard should reject them, but v3 predates that guard.
- 96 v3 rows (all `BTWUSDT`) have a Min60 cadence gap inside the prior 30-day
  feature horizon. Forward-window validation does not protect `change_30d`,
  rolling indicators or other feature history from this contamination.

Required before any rebuild/calibration:

1. return a typed fetch failure distinct from a genuine empty market window;
2. do not advance/persist a failed page; abort the symbol or retry it later;
3. validate cadence/coverage for every feature timeframe, not only the forward
   label window;
4. record explicit suspension/listing gaps separately;
5. invalidate/rebuild affected cached windows, then rebuild v3 and run the
   prescribed comparison triage.

No cache files or datasets were modified during this audit.

## Codex v3 quality-impact audit — 2026-07-26 03:46 MSK

The proposed forward guard was simulated read-only against all 5,182 rows in
`pump_dataset_v3.csv` and the current Min5 cache:

```text
passes current 90% / max-gap-12 policy: 5,163
rejected: 19 (0.367%)
rejected symbols: ASTEROID 8, SOXL 4, BTW 2, and five single rows
```

On this cache, every other forward window is exactly complete. Therefore
90%, 95%, 99% and even 100% coverage reject the same 19 rows; the max-gap gate
is doing the work. Removing those 19 is cosmetic for aggregate labels
(`mean mae_pct` changes by only `-0.000089`), though the individual rows remain
invalid.

Historical feature contamination is more consequential and is not covered by
the forward guard:

```text
rows with a Min60 gap inside prior 24h:   4
inside prior 7d:                         25
inside the 400-hour indicator tail:      57
inside prior 30d:                        96
```

All 96 are `BTWUSDT`. Their mean `mae_pct` is 0.5121 versus 0.1671 overall,
and their mean `n_averages` is 0.9479 versus 0.4325 overall. Excluding them
reduces overall mean MAE from 0.16707 to 0.16056. This does not prove their
outcomes are false; it proves their calendar-horizon features are wrong and
the affected rows are unusually influential.

Before calibration, validate each feature at its own required horizon. A gap
inside 24h/7d/30d or the indicator warm-up must either block the row or make
only the affected feature explicitly unavailable. Do not treat `tail(N)` rows
across a multi-day gap as N consecutive hourly bars.

## Codex decision-clock audit — 2026-07-26 04:16 MSK

The decision shift was not propagated to `hour_utc`.
`build_features()` still derives it from `event.ts`, which is the hourly bar's
open timestamp, while the executable decision is `event.ts + 3600`.

In `pump_dataset_v3.csv`:

```text
rows checked: 5,182
hour_utc equal to decision hour: 0
hour_utc exactly one hour behind decision: 5,182
crosses a UTC-day boundary: 149 rows (2.875%)
```

This may be either a naming error or a feature-clock error:

- if the intended feature is pump-bar seasonality, rename it explicitly to
  `signal_bar_open_hour_utc`;
- if it represents entry/execution time, calculate it from `decision_ts`.

Do not silently keep the ambiguous column across the rebuild. Add a boundary
test for a 23:00 signal bar whose decision occurs at 00:00 UTC.

## Codex 4h-semantics impact — 2026-07-26 04:46 MSK

The partial-versus-closed `rsi_4h` ambiguity is quantitatively material.
Recomputing the feature from only fully closed UTC 4h buckets for v3 gives:

```text
valid comparable rows: 5,141
decisions outside a 4h boundary: 3,857
median absolute RSI difference: 2.96
p95 absolute difference: 15.53
maximum difference: 52.90
rows crossing the configured 61.6 gate: 482 (9.376%)
partial passes / closed fails: 424
partial fails / closed passes: 58
```

The existing stored value is exactly the partial 4h RSI reconstructed from
hourly bars known at `decision_ts`; that is not future leakage by itself.
However, it conflicts with the new tests/comments declaring a closed-4h-only
contract and with the external `Hour4` confluence path, which now excludes a
forming bar.

Do not silently replace one with the other:

- closed-4h semantics require recalibrating `min_rsi_4h=61.6`;
- partial-4h semantics require an explicit name such as
  `rsi_4h_partial_at_decision` and a runtime snapshot proven to contain no
  information after the decision.

An end-to-end poisoned-forming-bar test must assert the actual RSI/gate output,
not merely reimplement the timestamp filter inside the test.

## Codex temporal-validation audit — 2026-07-26 05:16 MSK

The 48-hour labels require a purge/embargo at every chronological boundary.
No committed training or calibration path provides one:

- `ai/training/validate.py::chronological_split` slices rows at 70/15/15 with
  no time grouping, purge or embargo;
- `ai/training/train.py` reserves its validation and test frames but does not
  evaluate them; its train frame is split again inside `train_models()`;
- `ai/train.py` uses an 80/20 positional split and `TimeSeriesSplit(gap=0)`;
- `codex_trainer.py` also uses `TimeSeriesSplit(gap=0)`.

The pump CSV is not actually compatible with the first path as committed:
`validate_no_feature_leakage()` requires a `timestamp` column, while v3 has
`ts` and `decision_ts`. No committed consumer of `pump_dataset_v2/v3`, no
threshold-search script, and no reproducible implementation of the reported
"chronological test" was found. Therefore the claimed out-of-sample threshold
results cannot currently be audited from repository code.

Applying the existing positional splits hypothetically to the 5,182 v3 rows
(`SHA-256 4E62DDC2A70FBADC9E0AA4442CF6DB379D5C9EC955C82FC8F269FE78C48D5BE1`)
quantifies the missing embargo:

```text
70/15 train -> val: 130 / 3,627 (3.584%) train labels cross the boundary
15/15 val   -> test:  97 /   777 (12.484%) val labels cross the boundary
80/20 train -> test: 124 / 4,145 (2.992%) train labels cross the boundary
```

Same-symbol 48-hour label windows also overlap across those boundaries:

```text
70/15 train -> val: 280 overlapping pairs
15/15 val   -> test: 148 overlapping pairs
80/20 train -> test: 231 overlapping pairs
```

The 70% positional cut additionally splits one decision-time cohort itself:
31 rows at `2026-06-24 21:00 UTC` land in train and four rows with the exact
same decision timestamp land in validation. Across the full dataset, 4,259
adjacent same-symbol event pairs are less than 48 hours apart, so ordinary
row-wise `TimeSeriesSplit` folds are strongly dependent. A five-fold simulation
found `142, 79, 218, 96, 71` crossing train labels and
`159, 126, 324, 158, 115` same-symbol overlapping pairs by fold; folds 1, 3
and 5 also split equal-decision-time cohorts.

Before recalibration, commit a reproducible evaluation that:

1. groups all equal `decision_ts` values into the same partition;
2. purges at least the full 48-hour label horizon before each later partition;
3. states whether the embargo is global or per symbol (global is safer for
   shared BTC/regime features);
4. fits every threshold on train only, selects on validation, and reports test
   exactly once.

Until then, “held out of sample” in threshold comments is unsupported even
after the one-hour look-ahead fix.

## Codex feature-history causality audit — 2026-07-26 05:46 MSK

No additional direct post-decision OHLC read was found: the 1h/15m/direct-H4
filters, BTC frame, levels, fibs, divergence and liquidation-map inputs are
causal relative to `decision_ts`. There are nevertheless two material
sample-construction leaks and several clock defects.

First, the builder fetches exactly the requested 100-day event interval with no
feature warm-up (`ai/build_pump_dataset.py:50-66`). `build_features()` accepts a
row after only 60 hourly bars, while 7d/30d fields silently use whatever shorter
tail exists (`ai/pump_dataset.py:220-254`) instead of becoming unavailable:

```text
v3 rows with fewer than 169 prior hourly bars:  303 / 5,182 (5.85%)
v3 rows with fewer than 721 prior hourly bars: 1,516 / 5,182 (29.26%)
rows with fewer than 240 bars for fib context:   453 / 5,182 (8.74%)
rows with fewer than 400 indicator/level bars:   793 / 5,182 (15.30%)
```

A 60-bar probe returns the same 59-hour move under both `change_7d` and
`change_30d`, rather than `NaN`. Prefetch the longest feature warm-up before
the event interval, then enforce each feature's calendar horizon explicitly.

Second, historical events are selected using a present-day universe snapshot.
`SymbolUniverse.refresh()` filters current `amount24`/`riseFallRate`, sorts by
current 24h gain and applies the current cap before the builder fetches the
preceding 100 days (`universe.py:94-154`, `build_pump_dataset.py:39-51`).
Thus July listing/activity/liquidity determines which April events exist in
the sample. The magnitude cannot be recovered without point-in-time ticker and
contract snapshots, but this is selection/survivorship look-ahead by
construction. A calibration dataset needs date-versioned universe membership
or an explicitly unconditional contract-history source.

The ignored artifacts themselves show that membership is not stable or
reproducible: v2 and v3, written roughly 2.5 hours apart, share only 67
event-bearing symbols (21 v2-only, 22 v3-only; Jaccard 0.609). V1 and v3 share
only 26 (Jaccard 0.171). Without a saved universe/config/failure manifest this
cannot be separated into true ticker movement, different invocation arguments
or silent per-symbol fetch failures.

The static non-crypto exclusion is also incomplete. V3 still contains seven
`*STOCKUSDT` symbols and 227 rows (4.38%). A conservative obvious-equity/ETF
set including EWY, HK0700 and SOXL is 301 rows (5.81%); its resolution rate is
74.4%, versus 89.0% overall. These products demonstrably mix a different
regime into the claimed altcoin calibration.

Row counts are also used as clocks. One `BTWUSDT` v3 event described as a
six-bar run spans 587 calendar hours across a cache gap. Four 24-row
coin-return/pump-feature windows span 606 hours while their BTC comparison
still spans 24 hours. Cadence validation must happen before event detection and
every row-based feature, not just the forward label.

Funding uses `settleTime <= decision_ts`; 679 rows (13.10%) fall exactly on an
8-hour settlement boundary. This is safe only if the settled value is provably
available with zero publication latency. Record an as-of contract or lag it.

Robustness note: `_closed_by(empty)` returns a schema-less frame and `_rsi_at()`
then raises `KeyError("time")`. An empty Min15 response therefore discards the
entire symbol through the outer worker exception instead of recording a typed
data-quality failure.

## Codex threshold-provenance audit — 2026-07-26 05:46 MSK

The qualitative RSI correlation quoted by commit `fa7081f` is reproducible on
v2 (`rsi_15m=-0.17448`, `rsi_1h=-0.03831`, `rsi_4h=+0.06011` versus MAE).
The exact thresholds and claimed out-of-sample tail metrics are not.

`min_atr_pct=0.046` is not the stated 80th percentile of any full artifact:

```text
dataset   rank of 0.046   actual q80
v1           75.2852%    0.05319254
v2           76.1999%    0.05222107
v3           76.6113%    0.05082928
```

`min_rsi_4h=61.6` is near the 61st percentile in all three datasets. The exact
value may resemble one unknown slice, but no split/search artifact establishes
that derivation.

None of v1/v2/v3 stores the 12-bar liquidity value, so
`min_hourly_usd_volume=100000` is not reproducible from the CSV. Reconstructing
the runtime expression `sum(close*volume, 12)` from the current ignored cache
places 100,000 at ranks 43.44% (v1), 30.86% (v2) and 40.16% (v3), and the v1
ATR+liquidity filter leaves worst recorded drawdown 10.65 rather than the
commit's claimed 3.3. Raw volume units are themselves still unproven.

The threshold commits contain constants, prose and synthetic tests, but no
versioned raw snapshot, dataset hash manifest, liquidity feature, search grid,
split dates, purge, comparison JSON, expectancy implementation or generated
report. Treat the current thresholds as unverified hypotheses, not calibrated
OOS results.

## Codex replay input/math audit — 2026-07-26 05:46 MSK

Beyond the previously logged intrabar path and gap-stop problems, the replay
draft lacks an input contract:

- NaN entry passes validation and produces a NaN trade; infinite entry can
  raise `ZeroDivisionError`. `summarise()` then propagates NaNs into portfolio
  metrics.
- Rows with invalid high/low are silently skipped, yet the same invalid final
  row's close can be used for the horizon exit.
- No monotonic/unique timestamp check exists. Reversing the same two bars
  changed a probe from a one-leg +3% target to a two-leg +6% target.
- maker flags select only a fee tier; spread and taker-like slippage are still
  applied. Both maker flags false on a flat 100-to-100 trade still returns
  `-0.00057016`.
- `max_loss_on_deployed` is a gross mid-price stop, not a hard net-loss cap.
  With a 20% cap and default costs, a probe exits at `-20.0782%`.
- all-breakeven results report infinite profit factor instead of undefined.

Reject non-finite/invalid OHLC and costs, require sorted unique timestamps,
state maker execution semantics, and distinguish gross stop distance from
executable net loss before using replay output.

Focused validation after these read-only audits remains green:
`39 passed` for the dataset and replay suites (one pytest-cache permission
warning). No source, strategy, threshold, dataset, cache or runtime process was
changed by Codex.

## Codex MEXC volume-unit audit — 2026-07-26 06:16 MSK

The absolute liquidity gate is dimensionally wrong, not merely undocumented.
Official MEXC contract documentation gives:

- [`contractSize` as contract value](https://mexcdevelop.github.io/apidocs/contract_v1_en/#get-the-contract-information);
- [kline `vol` and `amount`](https://mexcdevelop.github.io/apidocs/contract_v1_en/#k-line-data);
- [ticker `volume24` and `amount24`](https://mexcdevelop.github.io/apidocs/contract_v1_en/#get-contract-trend-data).

Its BTC example has `contractSize=0.0001`, kline `close=33040.5`,
`vol=67332`, and `amount=222515.85925`:

```text
close * vol                         = 2,224,682,946
close * vol * contractSize          =       222,468.29 USDT
reported kline amount               =       222,515.86
raw close*vol overstatement         =         9,997.9x
```

The small adjusted difference is consistent with trade VWAP. The ticker
example likewise makes `amount24` consistent with quote turnover, while
`volume24` is a contract count.

`MexcContractClient` and `HistoryCollector` map `data["vol"]` directly to
`volume` and discard both kline `amount` and contract size
(`mexc_client.py:209`, `history.py:71-79`). The strategy then names
`sum(close*volume)` USD (`signal_generator.py:403-419`). Correct exact turnover
is kline `amount`; a close-price proxy must at least multiply by
`contractSize`. The error factor is symbol-specific `1/contractSize`, so
cross-symbol absolute liquidity is incomparable. Within-symbol volume-spike
ratios remain invariant to a constant contract size.

A read-only check using current public contract details for all 168 cached
Min60 symbols found 45 with `contractSize<1`, 47 equal to 1 and 76 greater
than 1. Across 363,676 cached 12-bar windows:

```text
raw close*vol >= 100k:                     212,768 (58.505%)
contract-size-adjusted proxy >= 100k:       299,309 (82.301%)
raw false passes / false fails:              11,841 / 98,382
latest window raw / adjusted passes:             104 / 162
```

Current metadata applied to historical bars makes this an impact estimate, not
an exact historical reconstruction. Exact repair is impossible from the
existing cache because `amount` was discarded. The earlier Min1/Min60 scale
ratio still demonstrates a horizon mismatch, but its `$100k` pass rates are
not USD and must not be used as calibration evidence. In contrast, using raw
`amount24` for the current USDT-contract universe band is dimensionally valid;
do not multiply that field by contract size again.

## Codex target-label consistency audit — 2026-07-26 06:16 MSK

`time_to_target_min` and `dca_resolved` do not measure the same target:

- `time_to_target_min` searches for `low <= original_entry * 0.97`;
- the DCA loop searches for `low <= blended_entry * 0.97`;
- MAE/MFE/good-MAE continue over the full 48 hours after the DCA path exits.

This produces the following artifact-level mismatch:

```text
dataset   DCA resolved but original target never hit
v1       211 / 5,523 (3.82%)
v2       148 / 5,042 (2.94%)
v3       324 / 5,182 (6.25%; 7.02% of resolved rows)
```

In v3, another 494 original-entry targets occur after the modeled blended-DCA
exit (median lag among these rows 227.5 minutes, maximum 2,710). Also, 1,489
rows report fewer DCA adds than the number of levels implied by their
full-horizon MAE because the trade-path loop stopped at exit while MAE kept
observing the market.

These are not one coherent outcome family. Either rename the existing field to
`time_to_original_entry_target_min` and explicitly retain fixed-horizon market
labels, or add a separate actual blended-DCA resolution timestamp and
trade-path MAE/MFE.

The DCA loop also imposes `high -> fill every add -> low -> resolve` inside
each 5-minute bar. A low-first alternative changes `(resolved, n_averages)` in
25 v3 rows, including the resolution flag in six. Choose a conservative
same-bar rule or use finer data; do not present the current order as observed
path.

## Night review closeout — 2026-07-26 06:46 MSK

The safe independent review scope is complete. Final repository state:

- Claude HEAD remains `8cc31fc`, one commit ahead of its tracking ref;
- no merge or push occurred;
- Claude's tracked draft is still only `ai/pump_dataset.py`; replay, its tests,
  E2E dataset tests and this handoff remain untracked;
- last focused dataset/replay run: `39 passed`;
- last stable full run after Claude's replay draft: `243 passed, 4 skipped`;
- no Python/trading process is running; live mode was never enabled.

The blocking order before any further threshold or strategy conclusion is:

1. Preserve MEXC kline `amount`, contract size, venue, closed-bar time and typed
   fetch/data-quality failures. Invalidate/refetch the lossy caches.
2. Define a point-in-time historical universe and persist its selection inputs,
   rejected symbols and invocation manifest.
3. Rebuild with a pre-event warm-up at least as long as the longest feature,
   strict cadence/alignment checks on every timeframe, and one explicit
   partial-versus-closed H4 contract.
4. Separate fixed-horizon market labels from executable trade-path labels;
   define stop/gap/same-bar/cost/capital semantics before calling the result
   expectancy.
5. Add decision-time cohort grouping plus a global 48-hour purge/embargo, then
   commit the calibration/search/report artifacts and run the prescribed
   comparison triage.
6. Only after those research inputs are reproducible, wire a public
   scan-only MEXC runtime with execution disabled and validate it on fresh
   closed data. Private/live execution remains out of scope.

Do not tune the present thresholds around these defects: the current artifacts
cannot support the claimed OOS profitability or tail-risk conclusions.

## Claude gate-calibration finding — 2026-07-26

Independent of the Codex threshold-provenance audit above, the same conclusion
was reached by measuring the runtime funnel directly. Recording the rejecting
layer over 32,996 hourly bars across 12 cached symbols with shipping defaults:

```text
layer1_pump_detection            25,818   78.25% of bars
layer1b_quality_gate              6,926   96.5% of what reaches it
layer1c_market_context               81
layer3_entry_location               162   95% of what reaches it
layer4_fake_filter                    1
layer_confirmation (pending/inval)    8
PASSED                                0
```

The earlier diagnosis blaming the confirmation bar was wrong. Confirmation only
sees eight candidates because two location gates have already removed the rest.

Both are calibrated on the wrong population. Measuring what each gate sees on
the bars that actually reach it:

| Gate | Threshold | Observed distribution | Pass rate |
|---|---|---|---|
| `min_atr_pct` | 0.046 | p50 0.0126, p95 0.0375, p99 0.0718 | 3.43% |
| `min_hourly_usd_volume` | 100,000 | p50 2.1e6 | 83.52% |
| `pump_entry_max_dist_from_peak_pct` | 0.015 | p10 0.0346, p50 0.0863 | 3.03% |

`min_atr_pct` sits at roughly the 96th percentile of the population it filters
and the distance gate at the 3rd. Both were fitted on the simplified event
population, where a 5% low-to-close move had already occurred by construction,
so high ATR and proximity to a fresh extreme were guaranteed by selection. The
liquidity gate is not binding.

Corroborates the Codex finding that 0.046 is not the claimed 80th percentile of
any artifact, from a different direction: percentile of the *runtime* population
rather than of the dataset.

Two changes landed at `f7352f2`:

- `pump_stop_buffer_pct` is now a floor under `stop_buffer_atr_mult * ATR`, and
  the confirmation invalidation level uses the same computation as the stop.
- `ai/runtime_dataset.py` records every gate's own measurement on each row, and
  `calibration_config()` opens the gates so thresholds can be fitted on the
  population they will run on. It removes the risk limits and is a measurement
  instrument only.

The 4h structural anchor was implemented and measured: the max over a 48h span
is the same number at 1h or 4h resolution (median difference 0.0000%, identical
for 88% of symbols). It is not a lever and is not presented as one.

Calibration population in progress with the Codex data-quality guards applied
(non-crypto proxies excluded per `DEFAULT_EXCLUDED_PATTERNS`; rows whose 120-bar
warm-up crosses a cadence gap rejected). Thresholds will be selected on replayed
P&L including fees, with symbol-clustered confidence intervals.

The bot remains stopped. No live execution, no credentials used.

## Claude no-edge finding — 2026-07-26, supersedes the expectancy claims above

The runtime calibration population was built (31,295 candidate rows, 278
symbols) and then reduced by the data-quality guards this file prescribes:
4,398 rows from 35 equity/commodity proxies and 10 rows whose 120-bar warm-up
crossed a cadence gap. 26,887 rows across 226 symbols and 133 days survive.

Thresholds were fitted on the first 60% by decision time, purged by the full
48h label horizon, and reported once on the remainder. On the DCA accounting the
result looked strong:

```text
test unfiltered: n=10189  mean +0.0134  CI [+0.0110,+0.0157]  win 83.0%
test rule:       n=  338  mean +0.0207  CI [+0.0162,+0.0250]  win 94.4%
walk-forward: 5/5 folds positive
```

That result does not survive contact with how the signals are used. It is P&L
per unit of *deployed* capital under a plan that adds up to six legs at 8%
steps. Averaging into a loser and exiting 3% below the blended entry produces a
high win rate and a positive per-trade mean by construction, which is the
martingale artifact this file warned about.

Replaying the same rows as a single entry - one position, one stop, no
averaging, which is how these signals are actually traded - removes the effect
entirely. Grid over targets 2/3/5/8% and stops 3-50%, all 28 combinations
negative, best -0.14%, clustered on the 0.217% round-trip cost.

The decisive control is a random entry on the same symbols over the same window:

```text
tp/sl       random entry              bot signal
3%/5%    -0.28% [-0.48,-0.08]     -0.29% [-0.95,+0.41]
3%/12%   -0.31% [-0.62,-0.02]     -0.31% [-1.29,+0.63]
5%/12%   -0.31% [-0.70,+0.08]     -0.14% [-1.34,+1.12]
5%/20%   -0.44% [-0.99,+0.14]     -0.35% [-1.81,+1.14]
```

The signals are indistinguishable from random entries. Both sit at minus the
transaction cost. Any true edge is smaller than roughly +/-0.5% per trade at
this sample size, against a 0.217% cost.

Uncensored adverse excursion before the 3% target, for the 338 rule-filtered
test signals: p50 3.1%, p90 27.0%, p95 45.7%, p99 118.0%, max 132.9%. At 100x
with a 2% position on cross margin, 4.44% of signals reach an account loss.

Retracted: the earlier +0.0208 expectancy, the "significantly loss-making
unfiltered / profitable filtered" split, and the portfolio result. All were
measured on DCA-deployed accounting or on the simplified event population.

What has never been tested is what the user actually asked for. In the current
signal path: Fibonacci is absent; `require_level_overhead` defaults False;
`weakness_layer_enabled` defaults False; the liquidation map reaches only the
chart, not the decision; `require_confluence` and `min_rsi_1h` are declared and
never consumed. What was measured and found edgeless is the generic
band-breakout + RSI + volume pump fade, not the technique set the strategy is
supposed to implement.

The bot remains stopped. No configuration in this repository currently has
demonstrated positive expectancy for a manually entered trade.

## Codex independent review of Claude's nine follow-up commits — 2026-07-28

Scope: commits `ead2aa1` through `9f71a86`, reviewed without changing strategy
logic, thresholds, branches, or runtime state. The safe operational conclusion
above remains unchanged: there is still no basis for live trading. The precise
research claims and the new scanner are not ready to rely on, for the following
reasons.

### P1 runtime and observation blockers

1. **The scanner evaluates forming candles.** `MexcContractClient.fetch_ohlcv()`
   returns the current MEXC bar and `app/scan.py:67-98` immediately uses it for
   the BTC benchmark and entry decision. `HigherTimeframeCache` does the same for
   the 4h RSI/structural anchor. A live probe during an hour returned the bar
   stamped at that hour's open. Offline code explicitly uses `_closed_by()`, so
   scanner and calibration semantics differ and an alert can repaint before the
   bar closes.
2. **Every new observation is backfilled with pre-signal history.**
   `app/scan.py:136` records `signal_bar_ts=None`; the tracker converts that to
   `last_bar_ts=0`, so the next `update_frame()` consumes all 320 bars supplied
   by the scanner. A probe immediately recorded historical min/max and both TP
   and SL hits. Repeated changes to the current forming bar are then skipped
   because its open timestamp has already been seen.
3. **One stateful strategy is shared by all worker threads.**
   `app/scan.py:105-106` calls one `LayeredPumpStrategy` concurrently.
   `SignalGenerator.last_diagnostics`, pending confirmation state, and
   `VolatilityContext` are mutable shared state. A deterministic two-thread
   probe produced cross-symbol metadata (`A -> A`, `B -> A`), so an alert or
   observation can receive another symbol's entry/SL/TP. The first volatility
   sweep is also still order-dependent because an empty frozen list falls back
   to the in-progress mutable observations.
4. **Required market context still fails open.** A BTC fetch failure sets the
   benchmark to `None`, and `core/signal_generator.py:357-363` treats the
   relative-strength gate as passed. The test at
   `tests/v2/test_scan_v2.py:123-127` currently codifies that unsafe behavior.
   HTF absence now fails closed, but an arbitrarily stale cached HTF frame is
   served indefinitely after later fetch failures.
5. **Delivery accounting is not delivery accounting.** The tracker sets
   `delivered=bool(alerters)` before Telegram is attempted. The Telegram wrapper
   ignores a `False` send result, so a failed HTTP delivery can be recorded as
   successful. Both LONG and SHORT intents are passed to `record_short()`.
   Hourly frames also cannot measure the tracker's advertised
   3/5/10/15/20-minute outcomes.
6. **No single-instance lock or persistent alert dedup exists.** Two scanner
   processes can send the same setup and concurrently replace the same
   observations JSON. This regresses the duplicate-process protection already
   present in the main runtime.

The scanner is genuinely signals-only and has no private MEXC/order-placement
path. That is a real safety improvement, but it does not make its alerts or
observations valid yet.

### Runtime-dataset parity blockers

- `calibration_config()` disables confirmation and opens several gates, so its
  rows are permissive pre-confirmation candidates, not delivered runtime
  signals. Calling their comparison with random a comparison of "bot signals"
  overstates what was tested.
- Indicators are computed once over the full historical frame and an
  ever-growing window is sent to the strategy. The scanner supplies only 320
  bars each cycle. Cumulative VWAP therefore differs after bar 320 and can change
  the Layer 4 result.
- `_GATE_FEATURES` requests fields that the trace does not emit
  (`level_dist_pct`, `msb_confirmed`, `sentiment_index`, `vwap_dist_pct`).
  HTF RSI and overhead measurements are disabled by the calibration config, so
  those columns are absent as well. The claim that every opened gate is still
  measured is false.
- The default scanner is Min60, but bar-count parameters retain minute-strategy
  meanings: `pump_window_bars=45`, `confirmation_max_wait_bars=3`,
  `msb_recent_bars=6`, and the comment describing a roughly 20-minute pump.
  They now mean 45 hours, 3 hours, and 6 hours. The new dataset measures this
  altered strategy, not the former Min1 technique.
- Runtime rows are labelled by the old arithmetic equal-quantity DCA
  `label_event()`, while `replay_short()` defaults to equal-notional legs. A
  one-bar probe at entry 100/high 108/low 100.8 resolves under `label_event()` but
  remains unresolved under the committed replay. Training labels and claimed
  executable PnL therefore implement different position-sizing contracts.
- `forward_window_quality()` still counts duplicate timestamps as coverage and
  accepts a completely off-grid shifted window. The 90% default also permits
  roughly 4.8 scattered hours to be absent from a 48-hour label. The later
  warm-up-gap rejection and proxy removal were out-of-band CSV operations, not
  committed builder logic.
- Proxy removal uses broad substrings. It removed 109 `FARTCOINUSDT` rows, 208
  `FILECOINUSDT` rows, and 313 `SPXUSDT` rows along with the intended stock/oil
  proxies: 630 crypto rows were classified as TradFi solely because their names
  contain `COIN` or `SPX`. The handoff's statement that all 35 removed symbols
  are equity/commodity proxies is therefore false and the filtered population is
  selection-biased.

### Reproducibility and DCA denominator recheck

The ignored local artifacts exist, but no committed command recreates them:

```text
runtime_calibration.csv
sha256 ef52628b1e50464098686020403eb62dc0b30b88c309ca8769943146728cabb4
31,295 rows / 261 symbols (not the documented 278)

runtime_calibration_pnl.csv
sha256 a56dff96b39108fd6be1f811d544e260fbb6e5a9a6ae5f86df7c308661fc23b1
26,887 rows / 226 symbols
```

The documented 10,189-row DCA test is reproducible from the second file by
sorting on `decision_ts`, taking row `int(26887 * 0.60)` as cutoff
`1778522400`, and testing rows with
`decision_ts > cutoff + 48 * 3600`. Its three denominators give opposite
impressions:

```text
mean(pnl_on_initial / realised_legs)       +1.3362%
symbol-cluster CI                         [+1.0911%, +1.5749%]

mean(pnl_on_initial)                       -0.6192%
symbol-cluster CI                         [-2.0023%, +0.5218%]

sum(pnl_on_initial) / sum(realised_legs)   -0.4575%
symbol-cluster ratio CI                    [-1.4438%, +0.3965%]
```

The stored `pnl` column is exactly `pnl_on_initial / legs`. Averaging that ratio
per trade gives one cheap one-leg winner the same weight as a capital-heavy
multi-leg loser and uses an ex-post denominator known only after the path. This
is the systematic pro-averaging bias behind the earlier positive number. Honest
capital accounting is statistically unresolved here; it does not establish a
positive DCA edge.

The 338-row rule cannot be reproduced: its predicates and fitted thresholds are
not recorded, and there is no versioned grid/random-control script, seed,
comparison JSON, manifest, or generated report. The single-entry/random table
was added only as prose. Overlap of two separate confidence intervals is not a
paired test of equality, and the statement that edge is bounded to roughly
`+/-0.5%` contradicts the displayed bot intervals reaching
`[-1.81%, +1.14%]`. The conservative statement is only that the current
evidence does not demonstrate edge.

### Replay and portfolio defects still affecting reported risk

- `backtesting/replay.py:169-185` exits a stopped trade before adding the stop
  bar to `worst_drawdown_on_initial`, so a one-bar stop reports zero drawdown.
  It does not read `open`; a gap is approximated from `low`, which can
  substantially understate the first executable buy-back price for a short.
- `backtesting/validation.py:152-182` settles positions in insertion order rather
  than `exit_ts` order. Reported max drawdown is therefore order-dependent and
  ignores open-position mark-to-market drawdown entirely.
- Funding, liquidation/margin mechanics, and a paired random-entry control
  remain absent. These modules are useful path diagnostics, not a complete
  executable-expectancy proof.

### Scope and tests

Commit `f7352f2` changes strategy defaults (`stop_buffer_atr_mult=0.5` and
`structural_anchor_htf=True`) and therefore changes eligibility, stop, target,
RR, confirmation invalidation, and signal count. That is a strategy change, not
an infrastructure fix, and should not be merged without an explicit user
decision.

Genuine fixes retained in the review: MEXC `amount`/`amount24` turnover units,
offline `+1h` decision alignment, HTF absence failing closed, next-bar wick
invalidation, interleaved DCA stop/add ordering, public-only scanner wiring, and
basic request pacing. They are incomplete in the edge cases above.

## Unified strategy/AI contract foundation — 2026-08-03

The current authoritative implementation plan is
`docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md`. It reconciles the five-layer
strategy, intended causal features, model roles and the single-position
contract after a fresh code/data-path audit.

Implemented in the working change associated with that plan:

- `ai/reversal/feature_contract.py` is the first versioned machine-readable
  registry (`mexc_reversal_features_v1`) with a stable SHA-256 contract hash,
  explicit feature roles/runtime statuses and null + observed-bit missingness;
- every MEXC population record now contains a fixed-schema causal
  `feature_snapshot`, including HOLD/error rows;
- point-in-time funding from the frozen universe snapshot is actually passed
  into the strategy instead of only being journalled;
- Layer 4 records funding/long-short source availability separately from numeric
  zero;
- fallback sentiment, missing volume-profile levels, missing overhead levels and
  turnover proxy values now carry explicit availability instead of becoming
  observed zeroes;
- the scanner fingerprints strategy config and universe policy, and the feature
  contract separates executable-schema and roadmap hashes;
- `ai/reversal/population_dataset.py` is the first strict consumer of the
  population journal. It rejects incomplete cycles and feature
  version/hash/schema drift, recomputes snapshots from source metadata, checks
  PopulationDecision timing/bar/status/IDs, exposes a role-whitelisted model
  input API and does not accept legacy event-conditioned CSV;
- `select_single_position()` no longer substitutes a filled runner-up after a
  top-score candidate is found unfilled using future price information.

This is foundation only. The trace is still stage-conditioned, scanner timing
is not yet executable-label timing, and the single-position replay is not yet
wired to the journal. No model may be trained or enabled from this partial
snapshot. Next priority is the Phase 1 time/spec contract described in the
master plan.

Validation of this foundation on the full MEXC tree:

```text
352 passed, 4 skipped, 2 known pytest collection warnings
```

Publication state for this foundation:

```text
0b010e8 fix(backtest): prevent hindsight cohort substitution
3ff8de0 feat(ai): add causal MEXC feature contract
29536f1 docs: unify MEXC strategy and AI roadmap
f0b43d6 docs: record AI foundation publication state
remote: origin/claude/codex-project-review-04581e (fast-forward)
```

### Historical 2026-07-28 independent-review validation

The following smaller counts belong to the earlier reviewed tree, not to the
current `f0b43d6` foundation:

```text
focused changed-area tests: 87 passed
full pytest: 287 passed, 4 skipped, 2 collection warnings
```

Green unit tests do not cover closed-bar parity, observation backfill,
cross-thread trace isolation, multi-bar confirmation gaps, actual Telegram
success, single-instance dedup, or reproduction of the published statistics.
Keep the scanner and trading bot stopped until those blockers are resolved and
fresh closed-data observation is repeated.

## Phase 1 slice 1 — causal time and cycle/cohort identity — 2026-08-03

Branch `claude/codex-project-review-04581e`, descending from the AI foundation
anchor `f0b43d6`. Scanner, external APIs and Telegram were not started; `.env`
was not read; no private, testnet or live path was touched.

### What was wrong

The independent re-entry audit confirmed two defects that block executable
labels regardless of any strategy question:

- `backtesting/single_position.py` grouped candidates by equality of a float
  `decision_ts`, so worker latency decided which candidates competed. The replay
  additionally required the first bar to open exactly at that per-symbol wall
  clock, which is unreachable in practice.
- `trading/market_data/universe.py` set `refreshed_at` before its own request and
  that value fed `cycle_id`, dating every ticker value earlier than the process
  could have held it.

### New temporal schema

```text
cycle_started_at            start of the scan pass
candle_cutoff_ts            last closed bar boundary at or before it
<source>.request_started_at when the request went out
<source>.received_at        when the response arrived   (never refreshed_at)
<source>.source_as_of       the causal cutoff of the data itself
ranking_ready_ts   = max(last per-symbol decision, last market response)
cycle_completed_ts = max(now, ranking_ready_ts)
actionable_ts      = max(all source received_at, ranking_ready_ts)
entry_eligible_ts  = max(actionable_ts, cycle_completed_ts)
entry_bar_open_ts  = first aligned bar strictly after entry_eligible_ts
```

All fail closed: a response before its own request, closed market data claiming a
moment after the answer, a cycle sealed before it was ranked, an unaligned entry
bar, an entry bar that does not follow the decision, or one that skips a bar the
decision could have reached.

### Changed identities

| Identity | Before | Now |
|---|---|---|
| `cycle_id` | bound `universe_refreshed_at` (pre-request) | binds `universe_received_at` |
| cohort | float `decision_ts` equality | explicit `cohort_id` on `EntryPlan` |
| entry reference | `plan.decision_ts` | `plan.entry_bar_open_ts` |
| `entry_ts` | decision instant | fill instant (the entry bar's open) |
| population journal | schema 1 | schema 2, reader fail-closed on 1 |
| single-position contract | schema 1 | schema 2 |

Wall-clock instants remain outside `input_hash`, so identical market inputs still
hash identically however slowly the scan ran.

### Commits

```text
a21a729 feat(time): measure when each market-data source actually answered
ce01c06 feat(journal): add the cycle envelope and causal timing schema v2
20d4b73 fix(backtest): group cohorts by explicit identity, not float decision_ts
bab837b test: prove cycle identity, entry timing and schema fail-closed
```

### Tests

```text
388 passed, 4 skipped, 2 known collection warnings
```

Baseline before the slice was `352 passed, 4 skipped`; the 36 new tests are the
nine required guarantees. Three consecutive full runs were identical, so the
parallel cases are not scheduling-flaky.

Two real defects were found by the new tests rather than by reading:
`ranking_ready_ts` could precede the last market response depending on thread
scheduling, and `entry_ts` was dated at the decision, which credited funding to a
position that was not yet open. Both are fixed.

### Not changed in this slice

Benchmark fail-open semantics, thresholds, strategy defaults, feature parity,
model training, the label builder and external dependencies are untouched.

### Still open

- P0 benchmark gate fails open at `core/signal_generator.py:358-362` while the
  higher-timeframe gate beside it fails closed. Fixing it changes signal counts,
  so it needs an explicit user decision and its own slice.
- P0 layer trace still stops at the first failed gate, so late features are
  structurally missing and the missingness mask correlates with the rule outcome.
- P1 base OHLCV and HTF share one `market_data` timing spanning the parallel
  pass; per-symbol and per-timeframe split is the next timing refinement.
- P1 raw contract ledger, instrument specs, cross-process journal lock, and the
  journal's refusal to reopen after a crashed tail all remain as recorded.

No edge is claimed or implied by this slice. It makes selection reproducible and
the entry reachable; it does not make the strategy profitable.

## Phase 1 rework after adversarial review — 2026-08-03

The first Phase 1 slice was not accepted: 388 green tests did not cover several
runtime and schema blockers. Sixteen findings were verified against the code and
fixed in three reviewable slices. Scanner, external APIs and Telegram were not
started; `.env` was not read.

### Retracted from the first attempt

- "Wall-clock timing does not change the snapshot hash" held only for
  `decision_ts`. The snapshot itself carried `universe_refreshed_at`, so timing
  did enter market identity.
- "An empty universe leaves an envelope" - the envelope was built but never
  persisted, so there was no durable evidence.
- "Worker-order acceptance complete" - premature. The cold-start volatility floor
  still depended on scan order.

### Slice A `8b03d59` - runtime and backtest correctness

The cutoff was derived after the universe refresh, so a refresh crossing a bar
boundary produced a cutoff later than the cycle start: a false provenance claim
and an envelope-invariant crash. It is frozen before the request now.

EntryPlan gained `entry_eligible_ts`, and the replay requires the entry bar to be
exactly the first reachable one - a later aligned bar silently measures a
different trade. ScoredCandidate now requires the result to match the plan on
decision_ts, entry_bar_open_ts and, when filled, entry_ts.

Two cohorts targeting one bar are resolved by which became actionable first, not
by cohort_id ordering, which is SHA order and causally meaningless. The bar is
consumed by the attempt, not the fill: whether the leader filled is knowable only
after that bar printed.

VolatilityContext tested its frozen list for emptiness, so a cold start fell
through to the live observations of the sweep in progress. A sweep is now an
explicit flag; proved on the real LayeredPumpStrategy over 28 symbols in both
orders.

### Slice B `577dd9d` - journal and envelope schema

The envelope was copied onto every decision row: quadratic in universe size, and
past 256 symbols the ordered universe exceeded the per-row collection bound, so a
300-symbol scan raised instead of journalling. It is now one header record, the
rows, then a footer with row count and a digest over the ordered snapshot IDs.
Journal schema moved to 3; two incompatible layouts must not both be called v2.

The writer refuses a file whose first record declares another version, a batch
mixing versions or carrying foreign rows, and a file whose tail lost its newline
- appending there would glue two JSON objects into one unparseable line. Empty
and error cycles are written and read back after a restart. The reader rebuilds
the envelope from its own fields and requires every cycle-level fact on a row to
equal it.

### Slice C `7290863` - source and feature time contract

Feature contract `mexc_reversal_features_v2`, executable hash
`20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c` pinned by
literal. The snapshot carries only `bar_cutoff_ts`; the universe response instant
moved to `feature_provenance`, which is excluded from `input_hash` and carries
the envelope hash so a row stays tied to the timing it was ranked under.

Sources are reported separately - universe ticker, contract details, benchmark,
base OHLCV, higher timeframe - each with its own request/response instants,
status and cache provenance. A cached ticker response keeps the instant the
exchange produced it instead of inheriting the current clock. `min()` clamping is
gone; an inverted clock fails closed.

The envelope declares `timing_basis="research_ranking_ready"`. It proves a
research replay may rank this cohort and enter on that bar. It does not claim a
live signals-only delivery could have reached it: record construction, the fsync,
the return path and the channel are unmeasured and any can cross the boundary.

### Tests

```text
423 passed, 4 skipped, 2 known collection warnings
```

Baseline before this rework was 388. Three consecutive full runs were identical.

### Still open

- P0 benchmark gate fails open at `core/signal_generator.py:358-362` while the
  higher-timeframe gate beside it fails closed. Out of scope by instruction;
  changing it changes signal counts and needs an explicit decision.
- P0 layer trace still stops at the first failed gate, so late features are
  structurally missing and the missingness mask correlates with the rule outcome.
  This is Phase 2 and blocks full worker-order acceptance.
- P1 base OHLCV and higher-timeframe timings are cycle aggregates over real
  per-symbol spans; per-symbol rows do not yet carry their own.
- P1 no cross-process lock on the journal; two scanners on one file still double
  append.
- P1 raw contract ledger, point-in-time instrument specs and the proposal/label
  bridge remain unbuilt.

No edge is claimed. This rework makes selection reproducible and the research
entry reachable; it does not make the strategy profitable.

## Phase 1 evidence hardening — 2026-08-08

This is the latest operational checkpoint. It supersedes only the open-state
verdict at the end of the 2026-08-03 rework; the historical findings above stay
as an audit trail. Work was performed only in the selected MEXC worktree on
`claude/codex-project-review-04581e`, starting from `59bb59c`. The three code
commits were pushed fast-forward and local/remote both reached `e0e4cb4` before
this documentation update:

```text
32e8fbe fix(journal): harden causal population evidence
0c32047 fix(backtest): bind replay outcomes to schema-v3 evidence
e0e4cb4 fix(strategy): fail closed without benchmark context
```

### Current version and identity matrix

| Boundary | Current version / identity |
|---|---|
| population journal | schema v4; default `data/runtime/mexc_population_decisions_v4.jsonl` |
| cycle envelope | schema v2 + canonical `envelope_hash` |
| reversal features | `mexc_reversal_features_v2`; pinned executable hash |
| market snapshot | `market_feature_hash(symbol, timeframe, snapshot)` |
| single-position | schema v3; plan/contract/replay-input/result SHA-256 |

`market_feature_hash` is the market-only identity. `PopulationDecision.input_hash`
and `snapshot_id` still bind the cycle and rule output (`action`, `reason`,
`confidence`), so separate persisted `MarketFeatureSnapshot`, `RuleEvaluation`
and `TradeProposal` entities are not yet complete. Wall clock is outside the
feature snapshot but still enters population identity through `cycle_id`; do not
claim that every row hash is timing-independent.

### Journal v4 and strict dataset boundary

- Header and footer both bind the exact `CycleEnvelope`; each feature-bearing
  row carries the same envelope hash.
- Completed cycles must contain exactly `envelope.universe_symbols` in order.
  `empty_universe` and `error` cycles must contain zero decision rows.
- Footer checksum covers every canonical field of every ordered decision row,
  rather than only snapshot IDs.
- Writer and restart audit reconstruct every `PopulationDecision`, re-derive
  `input_hash`/`snapshot_id`, rebuild the feature snapshot from source metadata,
  and verify symbol/timeframe-bound `market_feature_hash`.
- The reader validates the entire immutable file once before yielding its first
  cycle, then parses the same fingerprint again. A corrupt later cycle cannot
  leak an accepted prefix into an exporter/trainer.
- Duplicate cycles are rejected even in an A-B-A layout.
- A process-local path lock plus Windows `msvcrt` / POSIX `flock` sidecar lock
  protects `refresh -> dedup -> append -> fsync` across objects and processes.
- An incomplete/torn tail still blocks reopen. Recovery is intentionally manual:
  move the damaged file aside and start a new schema-v4 journal. Automatic repair
  is not implemented.
- If terminal-cycle persistence fails while journalling is enabled, scanner code
  raises instead of converting the attempt into an invisible gap.

### Source provenance and closed-frame semantics

- Universe ticker and optional contract details distinguish fresh response,
  TTL cache hit, stale fallback and first-request failure.
- `request_started_at`, `received_at`, `source_ts`, `cache_hit`, cache age,
  status and safe error code survive an exact envelope round-trip.
- A successful universe refresh uses its response instant as `refreshed_at`;
  stale fallback retains the prior successful TTL anchor so the next cycle can
  retry. Reused contract details are relabelled as cache data.
- `source_as_of` is absent when a failed source returned no data. Partial base or
  HTF availability is represented conservatively instead of claiming a wholly
  fresh cohort.
- Mixed fresh/cache higher-timeframe reads remain visibly cache-backed.
- `source_ts` for fresh ticker/details is currently the local response instant,
  not an exchange-supplied server timestamp.
- A closed OHLCV frame no longer makes an extra live ticker request. Its mark
  reference is the last closed close, matching the scanner's causal input.
- Base OHLCV and higher timeframe now have separate cycle timings, but still need
  per-symbol/per-timeframe provenance before executable label admission.
- The volatility sweep shares one frozen timestamp and cold-start distribution,
  removing scan-order drift from its floor.

### Single-position schema v3

The previous result object could be made internally plausible while changing
costs, sizing, exit timing or an opaque replay-input hash. The accepted contract
now requires:

- one SHORT entry, one stop, one TP and global concurrency exactly one;
- explicit fee, spread, slippage, equity/risk fraction, notional/leverage caps,
  quantity step/minimums, bar interval and maximum horizon;
- `plan_hash`, `contract_hash`, `replay_input_hash` and `result_hash` over
  canonical payloads;
- immutable `ReplayEvidence` containing the normalized gap-free horizon bars and
  strictly increasing, non-duplicated funding events;
- `ScoredCandidate(plan, contract, evidence, result)` with no legacy/default
  bypass;
- a fresh deterministic replay at the candidate boundary and again when the
  selector revalidates candidates. A random/rehashed input digest or a valid hash
  from different bars is rejected;
- exact entry/exit bar arithmetic, sizing, risk budget, fees and fill friction.

This is executable mechanics, not the finished label pipeline. Durable evidence
serialization/reader, forward-data manifests, point-in-time instrument rules and
the journal→proposal→label bridge remain Phase 1/3 work.

### Benchmark failure policy

The old BTC relative-strength gate failed open: missing benchmark data could
increase signals. `require_benchmark=True` is now the default. When relative
strength is active, missing, empty or non-finite benchmark context blocks L1c;
the previous behavior exists only behind explicit
`require_benchmark=False` for controlled ablation. `benchmark_available` is
recorded in the trace.

This deliberately changes eligibility/signal count. It is a correctness and
failure-semantics decision, not evidence that returns improved.

### Validation receipt

```text
529 passed, 4 skipped, 2 known PytestCollectionWarning (13.85s)
```

Focused adversarial coverage includes full-row/provenance partial edits and
corruption, restart
revalidation, A-B-A duplicates, concurrent spawned writers, fresh/cache/stale
scanner→journal→reader round trips, cost/sizing/timing forgeries with recomputed
result hashes, false replay-input hashes, evidence from different bars, reversed
funding and duplicate funding timestamps.

### Operational boundary and remaining order

- MEXC remains the target; the bot/scanner stayed stopped.
- No exchange API, Telegram, model training, testnet, private or live path was
  run. `.env` was not read. Historical credentials remain unrotated, so every
  private/live action is still forbidden.
- Root/Bybit was not changed; its three user-owned `.idea/*` modifications were
  preserved.
- Generic pump-fade still has no demonstrated stable edge after costs. No model
  has been fitted, promoted or enabled.

Next implementation order:

1. Finish `StrategySpecV2`: canonical configuration plus explicit
   feature/base/execution/15m/1h/4h intervals and physical windows.
2. Add per-symbol base/HTF provenance, point-in-time instrument rules and typed
   arm/confirm lifecycle.
3. Phase 2: compute gate-independent causal features and raw-contract ledger.
4. Only then build durable TradeProposal/OutcomeLabel and forward-data manifests.
5. Accumulate prospective population before logistic/rules/random baselines and
   the LightGBM + separate EV candidate. Repeat no-edge is an accepted result.

## Canonical StrategySpec and anchored journal v5 — 2026-08-08

This is the latest executable checkpoint and supersedes the remaining-order
statement immediately above. Work stayed inside the selected MEXC worktree.
The scanner, exchange APIs, Telegram, `.env`, model training, testnet and live
execution were not run.

```text
bebfd0d feat(strategy): define canonical MEXC strategy spec
2d0efcb feat(journal): chain schema-v5 population evidence
```

### Version and identity matrix

| Boundary | Current version / identity |
|---|---|
| MEXC StrategySpec | `mexc_strategy_v2`; dedicated `config/mexc_strategy_v2.yaml` |
| StrategySpec contract hash | `9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd` |
| committed default instance hash | `9f0b2d7035c2a82ab1b6d8595245b8c3a7a8b9faad17bea8c57f6fcacb189466` |
| population journal | schema v5; `data/runtime/mexc_population_decisions_v5.jsonl` |
| cycle envelope | schema v3; full canonical StrategySpec payload and identities |
| reversal features | `mexc_reversal_features_v2` |
| single-position replay | schema v3 |

The legacy root `config/config.yaml` is not a MEXC source of truth: it contains
different thresholds and Bybit-era settings. Production `app/scan.py` loads the
dedicated MEXC YAML once. Legacy CLI timeframe/candle arguments are assertions,
not post-hash overrides. Strategy, volatility context, base/benchmark requests,
HTF cache, indicators, volume profile and both history gates are constructed
from the same resolved object. `CycleEnvelope.from_dict()` rebuilds that object
from the persisted canonical payload and independently re-derives both hashes.

All numeric indicator fields now execute on both base and HTF calculations;
volume-profile minimum history/sample fields and the generator history gate are
also live inputs. The declared but still unwired `min_rsi_1h` and
`require_confluence` switches reject non-zero values instead of being silently
hashed and ignored. Defaults were preserved exactly: this is a configuration
and evidence migration, not a threshold calibration.

### Timeframe semantics

The current executable spec is explicit rather than aspirational:

- base and BTC benchmark: Min60, 320 closed bars;
- higher timeframe: Hour4, 120 closed bars, 30-minute request TTL;
- windows retain fixed counts of their source bars.

Consequently the current 45-bar pump window is 45 hours, confirmation wait is
3 hours, recent-MSB window is 6 hours, relative-strength lookback is 24 hours,
and the 12-bar structural HTF anchor is 48 hours. These durations preserve the
previous scanner exactly. They do **not** prove that 45 hours is the intended
fast-pump horizon. Changing to a faster physical window must be a separately
versioned strategy/research decision with a new instance hash and evaluation.
No 15-minute or execution-timeframe feed was invented in this refactor.

### Journal v5 trust contract

Each new file receives a random 256-bit `journal_id`. Cycles have contiguous
`sequence_no`, a domain-separated genesis, `prev_cycle_commit`, and a footer
`cycle_commit` over the exact canonical header, every full ordered decision row
and the footer core. Restart validates the complete chain present in the file.
A stale writer may adopt only an exact extension of the prefix it previously
observed; rollback, fork and rewrite relative to that cached prefix fail closed.

`PopulationJournal.checkpoint_receipt()` returns a detached receipt containing
the journal ID, sequence, cycle ID/commit and SHA-256/length of the exact raw
prefix. Returning or storing that unsigned receipt beside the journal does not
make it trusted. A caller must preserve it in an independently protected domain
(for example a pushed Git checkpoint or an authenticated immutable store) and
pass it explicitly to `verify_population_journal()`/the dataset reader.

The boundary is deliberate:

- v5 detects corruption, torn or incomplete writes, partial edits, splicing and
  a changed earlier cycle whose successor was not rebuilt;
- a fresh unanchored reader accepts a clean shorter valid prefix, so suffix
  deletion/rollback is detected only relative to a stale writer's cached state
  or an explicitly supplied trusted receipt covering the removed prefix;
- an actor able to coherently rebuild an entire **unanchored** file can still
  produce another internally consistent chain;
- an earlier trusted external receipt detects a rewrite of its covered prefix;
- data after that receipt is validated but remains unanchored and is excluded
  by checkpointed readers unless the caller asks for the tail explicitly;
- `model_input_records()` requires a trusted receipt by default. The only
  bypass is the explicit `allow_unanchored=True` research override.

The reader validates the whole file before export, then checks each second-pass
cycle ID/commit against the first pass before yielding. This closes the
same-size/same-mtime replacement race found during adversarial review. File
identity also includes device, inode, size, mtime and ctime.

### Validation and operational verdict

```text
full pytest: 576 passed, 4 skipped, 2 known PytestCollectionWarning (15.12s)
focused StrategySpec/runtime review: no remaining P0/P1
focused journal/checkpoint red-team: no remaining P0/P1/P2
git diff --check: clean
```

No trading edge was established and no model was fitted. The unkeyed chain is
not a substitute for an external anchor, and its contract hash is a pinned
declarative/version discipline rather than a hash of Python implementation
bytes. Exact default runtime parity is enforced by regression tests.

Next implementation order:

1. Choose and version the intended physical fast-pump windows; do not retune
   them inside a mechanical refactor.
2. Add typed arm/confirm lifecycle, per-symbol base/HTF provenance and
   point-in-time instrument rules.
3. Phase 2: compute gate-independent causal features and the raw-contract
   inclusion/exclusion ledger.
4. Build durable TradeProposal/OutcomeLabel plus forward-data manifests and
   connect them to single-position v3.
5. Only after prospective maturation run logistic/rules/random/no-trade
   baselines and then the LightGBM + separate EV shadow candidate.

## Independent Claude audit follow-up — 2026-08-09

The follow-up audit found no remaining P0 or P1 in the reviewed
StrategySpec/journal-v5 checkpoint. The only code change admitted from that
review is the test-only compatibility and behavioural lock in:

```text
258c35f test(strategy): pin v2 behavioral semantics
```

### P2 disposition

- **P2-1 — closed.** Golden vectors now pin the executable indicator modes,
  volume-profile levels and one representative `layered_pump_signal_v1`
  armed-to-confirmed decision/proposal trace (floats normalized to 12 decimal
  places for its digest). A frozen JSON fixture additionally retains
  the canonical `mexc_strategy_v2` payload, its exact contract/instance hashes
  and a full `CycleEnvelope v3` round trip.
- **P2-2 — no documentation change required.** The durable project topology is
  the root/Bybit line plus the selected MEXC line. Generated
  `.claude/worktrees/agent-*` and `context-*` worktrees are ephemeral. The review
  prompt already requires `git worktree list --porcelain`, says other worktrees
  may exist, and requires reporting them without modification; pinning generated
  names would make the handoff stale rather than safer.

### Guard boundaries and remaining caveats

- The frozen v2 fixture is a migration tripwire, not a v3 implementation. A
  future StrategySpec v3 must add explicit version dispatch/backward reading for
  existing v2 evidence; updating or regenerating the frozen fixture to make a
  bump pass would defeat the guard.
- The golden vectors preserve current behaviour, not the intended physical
  trading horizon. With the frozen default spec, 45 base bars are still 45 hours
  on Min60. Any faster hypothesis requires a new version/instance hash and a new
  chronological evaluation, not an in-place threshold/window rewrite.
- The tests do not choose a model estimand. Prediction at arm time and prediction
  after confirmation/proposal answer different causal questions and expose
  different information. The scoring instant, eligible population, outcome and
  payoff definition must be frozen before prospective labels or model fitting.

Latest validation at `258c35f`:

```text
full pytest: 580 passed, 4 skipped, 2 known PytestCollectionWarning (14.99s)
git diff --check: clean
```

No exchange/network call, scanner, bot, Telegram, model training, testnet,
private or live path was run for this follow-up, and `.env` was not read. These
locks establish compatibility and implementation identity only: they do not
establish trading edge, validate a model or relax any live-risk boundary.

## Version-dispatched StrategySpec evidence checkpoint — 2026-08-09

This append-only checkpoint supersedes the remaining v3-readiness caveat in the
previous section. The executable change is:

```text
1971b77 feat(strategy): preserve versioned evidence compatibility
```

### Frozen evidence and identity boundaries

- `mexc_strategy_v2` now has an explicit registered evidence decoder. Its
  parser, contract-hash derivation and instance-hash derivation remain bound to
  v2 rather than to a mutable global "current version" alias.
- The pinned v2 contract and default-instance hashes are unchanged. The frozen
  payload/CycleEnvelope fixture and behavioural vectors remain compatibility
  tripwires; no fixture was regenerated to conceal drift.
- `CycleEnvelope v3` dispatches by the persisted version, rebuilds the canonical
  payload and checks the exact `(version, contract hash, instance hash)`
  identity. Unknown versions and outer/payload version disagreement fail closed.
- Population journal v5 binds each file to one exact StrategySpec identity.
  Restart and append reject a different identity before export or additional
  bytes, including a different operator-selected YAML instance under v2.
- The strict population reader independently enforces the same homogeneous
  identity. `model_input_records()` carries the three identity fields as
  provenance metadata outside the feature vector, so downstream datasets can
  detect accidental cross-version or cross-instance concatenation without
  teaching the model the strategy identity.

This prepares a safe namespace boundary for a future `mexc_strategy_v3`; it
does not define v3 parameters, select Min15, change the frozen Min60/45-hour v2
control or choose an arm-time versus post-confirmation estimand. A future v3
must register separate types/parser/config/evidence while the v2 decoder and
fixtures remain readable and green.

### Independent validation and operational boundary

```text
full pytest at 1971b77: 590 passed, 4 skipped,
                       2 known PytestCollectionWarning (18.95s)
independent red-team: no P0/P1/P2 in the checkpoint change scope
git diff --check: clean
```

The checkpoint changes evidence decoding, identity validation and export
provenance only. It does **not** change strategy behaviour, indicators,
thresholds, network use, model selection/training, signal delivery, execution or
live-risk permissions. No exchange/network call, scanner, bot, Telegram, model
training, testnet, private or live path was run, and `.env` was not read.

Next implementation order remains:

1. Add typed arm/confirm lifecycle plus per-symbol base and
   per-symbol/per-timeframe HTF provenance.
2. Separate persisted identities for `MarketFeatureSnapshot`,
   `RuleEvaluation`, `TradeProposal`, `OutcomeLabel` and `ShadowPrediction`.
3. Add point-in-time instrument rules: contract size, quantity step, minimums,
   leverage rules, source timestamp and content hash.
4. Compute gate-independent causal features and the raw-contract
   inclusion/exclusion ledger.
5. Build the journal-to-TradeProposal-to-OutcomeLabel bridge and connect it to
   the single-position replay.
6. Compare physical-time hypotheses chronologically while retaining v2 as the
   frozen control; choose no faster strategy without evidence.
7. Only after prospective labels mature, run simple baselines and then the
   LightGBM proposal-conditioned outcome plus separate EV candidate, with
   CatBoost as challenger.

## Typed lifecycle and journal-v6 evidence checkpoint — 2026-08-11

This append-only checkpoint is the authoritative executable state after the
version-dispatched StrategySpec section above. It supersedes earlier statements
that per-symbol base/HTF provenance and typed arm/confirmation lifecycle were
still entirely missing. It does not supersede the no-edge verdict, the model
training prohibition or any live-risk boundary.

### Executable commits and validation receipt

The complete evidence series descends from `1971b77` and includes:

```text
eb238b2 test(journal): freeze schema v5 evidence boundary
c541eea feat(evidence): define candidate lifecycle contract
a604668 feat(evidence): bind exact closed-frame provenance
8569471 fix(evidence): reject malformed empty frame reads
cf6bc01 feat(strategy): emit typed candidate lifecycle evidence
9ef6b4f fix(evidence): finalize typed lifecycle semantics
bb1ca13 feat(journal): persist typed population evidence in schema v6
```

`bb1ca13` is the latest executable code tip for this checkpoint. Validation on
that committed source tree:

```text
full pytest: 723 passed, 4 skipped, 2 known PytestCollectionWarning (17.80s)
git diff --check before documentation update: clean
```

The two warnings are the existing collection warnings; no new warning class was
introduced by this series.

### Current version and identity matrix

| Boundary | Current version / identity |
|---|---|
| MEXC StrategySpec | frozen `mexc_strategy_v2` control |
| CycleEnvelope | schema v3 |
| causal cycle identity | version v5, unchanged by the journal-format bump |
| population journal writer | schema v6 |
| default population path | `data/runtime/mexc_population_decisions_v6.jsonl` |
| frozen population compatibility | schema v5, read-only, exact fixtures retained |
| reversal feature contract | `mexc_reversal_features_v2`; `20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c` |
| frame provenance | v1; `f4004ac933cc1725b2560e93ffbe278c826910424e350059ad29420ed3665dbf` |
| candidate lifecycle | `candidate_lifecycle_v1`; `cc75c871b7097aa215f9ac88c736b6572e2443318cb0cf9f8bdaf1b0c8cc8551` |
| single-position replay | schema v3, still offline and not yet connected to population labels |

`SCHEMA_VERSION` deliberately remains the public alias for immutable journal
v5 evidence, while `CURRENT_WRITE_SCHEMA` is 6. `CYCLE_IDENTITY_VERSION` remains
5, so moving the serialization format to v6 does not silently rename an
otherwise identical causal cohort. Existing v5 fixture bytes, commitment
domains and readers remain pinned. A v5 file can be verified and read, but any
attempt to append fails closed and directs the caller to a separate v6 file.
Model export from v5 additionally requires explicit `allow_legacy_v5=True`;
legacy evidence is never silently treated as typed v6 evidence.

### Exact closed-frame evidence consumed by the strategy

The typed strategy entrypoint now receives three explicit `FrameRead` objects:
base OHLCV for the candidate symbol, the BTC benchmark frame and the candidate
symbol's higher-timeframe frame. It no longer obtains benchmark or HTF input
from mutable shared strategy/cache state. After taking owned frame copies it
revalidates the exact source evidence and recomputes base indicators from the
raw base frame used by the decision.

Each `SourceReadEvidenceV1` binds the source and venue identities, canonical
venue symbol, timeframe, requested cutoff, request/receipt timing, expected
closed boundary, row count, first/last bar boundaries and canonical frame hash.
The latency-free market/frame identities of the three items are bound again by
one canonical raw-frame-bundle hash; request, receipt and cache facts remain in
their separately persisted operational evidence. The benchmark/base/HTF
bar-source timings in the CycleEnvelope are derived from that evidence rather
than accepted as an unrelated caller assertion. Universe and
allowed contract-details timings remain separate cycle-level provenance and are
not represented as `FrameRead` objects. Receipt times later than the decision
and row/envelope symbol, venue, timeframe or cutoff drift fail closed.

The evidence vocabulary distinguishes `fresh`, `stale`, `no_rows`,
`request_failed` and `not_requested`. A stale frame keeps its exact range/hash
and safe reason instead of becoming a fresh observation. An `evaluated` row
requires fresh base evidence; an entry action additionally requires benchmark
and HTF data through their expected closed boundaries. Empty, failed and stale
reads therefore remain auditable without being promoted into executable entry
evidence.

### Typed arm, confirmation and proposal evidence

`CandidateArmV1` binds the exact StrategySpec identity, raw input bundle,
symbol/side/timeframe, arm bar and cutoff, observed high/low/close, effective
invalidation level, confirmation policy and immutable arm trace. Its semantic
candidate ID excludes scheduling, persistence and delivery wall clocks.

Follow-up events use the explicit states `same_bar`, `waiting`, `confirmed`,
`invalidated` and `expired`; confirmation-disabled evaluation uses the separate
terminal state `bypassed`. Same-bar evidence must be an exact repeat of the arm
observation. Later observations must advance on the base timeframe with
monotonic observation counts and price-state semantics consistent with the
candidate side. Every event names its exact predecessor, state epoch and
confirmation observation identity.

A proposal observation is either not evaluated, created or rejected. It can be
evaluated only from a confirmation observation or from the explicit arm-bypass
path. A created proposal is bound to the exact input bundle, reference bar,
cutoff and decision reference close used to create it. These records prove what
the rules observed and proposed; they are not orders and do not yet constitute
the durable point-in-time `TradeProposal` needed by the replay/label bridge.

The scanner passes the exact base/benchmark/HTF triplet into this typed strategy
path and persists the resulting lifecycle event in the same population row.
Typed evaluation is transactional: malformed evidence cannot partially commit a
pending lifecycle transition. The legacy `generate()` path and its pinned v2
behaviour remain compatible; v6 scanner evidence uses the typed path.

### Journal v6, lifecycle chain and runtime ownership

A v6 cycle header persists the evidence-contract identities and exact benchmark
source evidence. Every decision row persists exact base and HTF evidence, the
raw bundle hash and an optional typed lifecycle event. The strict reader
reconstructs causal row identities and snapshots, checks the exact
row-to-envelope projection and admits only `HOLD`, `SHORT_ENTRY` or
`LONG_ENTRY`. Entry rows require a valid typed lifecycle outcome and current
source evidence. The existing journal commitment/checkpoint trust boundary is
unchanged: the internal chain detects accidental corruption and inconsistent
edits, while rollback or a coherent rewrite of an unanchored file still needs an
externally protected receipt to detect.

Across ordered v6 cycles the reader/writer validates witnessed lifecycle
predecessors and rejects duplicate candidates/events, orphan follow-ups and
forks. A new initial event for the same symbol right-censors the older active
candidate. This is evidence-chain validation, not runtime restoration: after a
process restart the scanner still does not rehydrate a pending in-memory
candidate. A deliberate restart right-censor record or a tested rehydration
contract remains pending.

The scanner now takes two complementary ownership guards. A process-local
whole-sweep lock serializes mutations of strategy confirmation state. A
non-blocking journal lifetime lock spans the scanner loop across threads and
processes, so another scanner using the same journal fails before its first
market request. Journal append locks still protect byte-level refresh, dedup,
append and fsync; they are not used as a substitute for runtime ownership.

Strict model export still requires a trusted external checkpoint unless the
caller explicitly selects the unsafe unanchored research override. StrategySpec
identity, exact benchmark/base/HTF evidence, raw bundle identity, lifecycle
event and typed-evidence status are returned as top-level partition/evidence
metadata. They stay outside the numeric `features` mapping and must not become
predictors merely because the exporter carries them beside a row.

### Unresolved gates and operational truth

- **No edge:** generic pump-fade still has no demonstrated stable positive edge
  after costs. Earlier positive-expectancy/DCA claims remain retracted.
- **No model:** no model has been fitted, selected, promoted or enabled.
- **Training remains blocked:** rule trace and feature missingness are still
  gate-conditioned. The current population can validate evidence mechanics but
  is not an admissible unbiased training population.
- **No point-in-time instrument/proposal-label bridge:** contract size,
  quantity step, minimums, leverage rules and their source timestamp/content
  hash are not yet frozen into the decision. A durable `TradeProposal` and
  `OutcomeLabel` connection to single-position replay is still absent.
- **Restart semantics remain incomplete:** the historical chain is validated,
  but pending candidate state is not rehydrated and an explicit restart
  right-censor policy is not yet persisted.
- **Secrets remain unsafe:** historical credentials have not been rotated.
  `.env` was not read. Private APIs, Telegram, testnet and live execution remain
  forbidden.
- **No operational run:** this series did not start the scanner/bot, call an
  exchange or other network service, train a model, deliver an alert or expose
  capital.

### Required next implementation order

1. Freeze and test the restart boundary: either persist an explicit
   right-censor event before accepting a new candidate or implement deterministic
   pending-candidate rehydration from externally anchored v6 evidence.
2. Persist separate versioned identities for `MarketFeatureSnapshot`,
   `RuleEvaluation`, `StrategyProposal` and `ShadowPrediction`; add point-in-time
   instrument rules including contract size, quantity step, minimums, leverage,
   source receipt and content hash.
3. Compute the full causal feature snapshot independently of early rule exits and
   add the raw-contract inclusion/exclusion ledger. Gate results belong in
   outcomes/metadata, not in the feature-availability pattern.
4. Build the journal-to-TradeProposal-to-OutcomeLabel bridge against
   single-position schema v3 with exact entry eligibility, one stop, one TP,
   sizing, fees, spread, slippage, funding and concurrency one.
5. Accumulate an anchored prospective runtime population, then run purged
   chronological rules/no-trade/random/logistic baselines. Only after those
   gates pass may LightGBM plus a separate EV head be evaluated in shadow/paper
   mode, with CatBoost and sequence models as later challengers. Another no-edge
   result must be accepted without threshold or model shopping.

## Final-product roadmap approval checkpoint — 2026-08-15

This documentation-only checkpoint promotes
`docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md` from reviewed draft to the
authoritative final-product roadmap. It does not alter, reinterpret or migrate
the frozen `mexc_strategy_v2` runtime/evidence contracts. The prior
`docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` is preserved as their historical
and executable-v2 audit source.

### Approval base and artifacts

- Branch: `claude/codex-project-review-04581e`.
- Review base: local HEAD and upstream both `ad30b02` before the S1
  documentation changes.
- Final-product roadmap: `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md`.
- Independent review prompt/receipt source:
  `docs/CLAUDE_FINAL_BOT_PLAN_REVIEW_PROMPT_2026-08-14.md` plus the supplied
  2026-08-15 closure verdict.
- Supporting S1 documents are
  `docs/ADR_MEXC_V3_FINAL_BOT_2026-08-15.md` and
  `docs/MEXC_V3_PREREGISTRATION_SKELETON_2026-08-15.md`; they do not mutate v2
  codecs, fixtures or evidence namespaces.
- No future publication commit hash is asserted in this checkpoint. Discover
  the eventual documentation-only commit from Git history.

### Independent closure verdict

Claude independently re-read the corrected master and returned
`APPROVE_AS_AUTHORITATIVE`; remaining P0/P1: **none**. The closure pass verified
that:

1. `actionable_delivery_at` is tied exactly to a durable per-channel receipt
   and the delivery SLA is therefore falsifiable;
2. journal v6 is described as the current writer for the frozen v2 line and is
   closed to v3 semantics;
3. P4/P5 contracts may be researched after P0 but cannot be accepted before
   P1–P3, and their schemas/population definition cannot be frozen before P3
   proves acquisition feasibility;
4. U15 requires an explicit narrowed-universe policy or stopping v3 when
   full-universe Min1 acquisition is not achievable under the frozen P3 gate.

### Scope and operational receipt

- Documentation only; no production code or frozen fixtures were changed.
- No scanner, bot, model, Telegram, private API or exchange process was run.
- No network call was made and `.env` was not opened.
- No pytest run was needed for the read-only document closure review; the last
  executable validation receipt remains historical and must not be presented as
  validation of future v3 code.
- No edge, model promotion, public-data pilot, private/testnet/live access or
  capital exposure is authorised by this approval.

### Next gate

This bounded documentation checkpoint is S1; discover its eventual publication
hash from Git rather than embedding a self-reference. The next code gate is S2
strict local history collection plus S3 aggregation contracts without network
access. A public Min1 pilot still requires separate explicit user approval after
S1–S3; model work remains blocked until an admissible population, causal labels
and preregistration exist.

## Strict history and deterministic Min1 aggregation checkpoint — 2026-08-15

This append-only checkpoint supersedes only the S1 “next code gate” immediately
above. It does not alter frozen v2 codecs, fixtures, runtime behavior or the
no-edge verdict. S2/S3 are local data-contract foundations; they are not a
public-data pilot, an admissible population, a v3 strategy or proof of edge.

### Publication chain and exact identities

```text
2a14299 docs(roadmap): approve MEXC final-bot contract
0ff1b3a feat(data): add deterministic Min1 aggregation contract
36e1446 feat(data): add strict MEXC history contracts
```

- `mexc_min1_aggregation_v1` contract hash:
  `0d851b253cde913d95a693e0db7296b59ff78a6048bacb20838386f2e8e20a21`.
- `mexc_strict_history_v1` contract hash:
  `6c17bd9de3e25210139da4491a1f35fbd0cec557707fb5d376a60ce23e04c6c1`.
- Files added by S2/S3:
  `trading/market_data/min1_aggregation.py`,
  `trading/market_data/strict_history.py`,
  `tests/v3/test_min1_aggregation_contract_v1.py`, and
  `tests/v3/test_strict_history_contract_v1.py`.
- Legacy `HistoryCollector`, existing MEXC feed/client behavior, `data/history`,
  bar/frame-provenance contracts, v2 specs and frozen fixture bytes were not
  edited or migrated.

### What S2/S3 now prove

- The strict collector has no default network transport. It requires an
  explicit half-open aligned range, injected transport and explicit storage
  root; it never converts network/HTTP/JSON/API/payload failure, truncation or a
  gap into valid no-data.
- Exact raw bodies and per-attempt receipts are persisted before parse;
  normalized completion requires the entire closed UTC grid, exact quote
  `amount` turnover, immutable raw/attempt/normalized artifacts and a final
  manifest whose source graph is reverified before publication.
- Min1 aggregation accepts only contiguous canonical UTC rows and exact
  per-row receipts. It emits complete epoch-aligned Min5/Min15/Min60/Hour4
  groups using deterministic first/max/min/last/`fsum` semantics.
- The S2→S3 adapter binds each consumed normalized row to its exact raw page and
  the complete S2 manifest. A coherently changed frame with reused receipts is
  rejected rather than laundered into derived evidence.
- Existing ignored `data/history` CSV files remain legacy/discovery-only and
  contain no admissible Min1 evidence.

### Verification and independent review

```text
focused tests/v3: 91 passed
full pytest:       822 passed, 4 skipped,
                   2 known PytestCollectionWarning (22.94s)
independent reviews: P0 none; P1 none; APPROVE S2/S3 contract checkpoint
```

No operational scanner/bot runtime, model, Telegram, private API or exchange
request was started. No public endpoint was contacted, `.env` was not read, and
no pilot data or runtime artifact was produced. Root/Bybit remained limited to
the three pre-existing user `.idea/*` changes.

### Open P2 and next gate

Before any U5 network authorization can be exercised, a bounded fake-transport
pre-pilot hardening slice must add or pin:

1. the official Futures endpoint/domain fixture without silently changing the
   legacy client default;
2. raw-response, page/row, storage and runtime budgets;
3. pacing/backoff/`Retry-After` and oversized-response tests;
4. a strict disk loader/restart reconciliation that verifies the complete
   raw/attempt/normalized/manifest graph and redundant raw byte lengths;
5. documented Windows best-effort directory-fsync/atomic-publication behavior;
6. exact sub-microsecond timing-tolerance disposition, safe public
   `Date`/quoted-`ETag` handling, and consistent typed storage errors for damaged
   duplicate-key/non-finite JSON artifacts.

Only after that local slice and an immutable pilot run manifest may the user
grant U5 explicitly. U5 is still **not granted**. The next permitted work is
local pre-pilot hardening; threshold search, v3 runtime, model fit, Telegram,
private/testnet/live access and capital exposure remain blocked by the master
plan gates.

## Bounded transport and restart-safe strict-history v2 checkpoint — 2026-08-16

This append-only checkpoint supersedes the “Open P2 and next gate” immediately
above. It hardens the per-range S2 acquisition boundary; it does not execute or
approve P2, change frozen strict-history-v1/aggregation-v1 or frozen
StrategySpec/journal-v2-control evidence, select v3 thresholds, or prove edge.

### Commit chain and exact identities

```text
ba8ea00 feat(data): add bounded MEXC futures transport contract
f8a6b5b feat(data): add restart-safe strict history v2
```

- Endpoint candidate `mexc_futures_kline_endpoint_candidate_v1`:
  `54f57d755cd679eb92444d48b38013621caad37067125e80cf7c5e45fe2ab220`.
- Per-shard limits `mexc_history_resource_limits_v1`:
  `937d053e33c513d128389259e308156c8758e5cfe44b5849e3eb27ea49d96bdc`.
- Retry policy `mexc_history_retry_policy_v1`:
  `78f92d14cc26ead1a372d840a05fe8a60dae97d5d9a3cdacc539a098194a2cc9`.
- Raw transport `mexc_futures_raw_transport_v1`:
  `7d3bd40c6753e7bda2f1904ce2ffa2ff55770ecce9ba6d5614d2b30ae0664d22`.
- Strict collector/storage `mexc_strict_history_v2`:
  `cce9922317ec5f0008f3b293103f9f5a17504b7143f81af1845d9d4765c44086`.
- Frozen `mexc_strict_history_v1` and `mexc_min1_aggregation_v1` remain pinned
  to `6c17bd9de3e25210139da4491a1f35fbd0cec557707fb5d376a60ce23e04c6c1`
  and `0d851b253cde913d95a693e0db7296b59ff78a6048bacb20838386f2e8e20a21`;
  their source and fixtures were not changed.

### What the checkpoint proves

- The fixture pins `https://api.mexc.com/api/v1/contract/kline/{venue_symbol}`
  but explicitly records `candidate_not_u5_verified`; both current-official-doc
  and live-endpoint verification are false. There is no real/default network
  executor. Tests use an injected fake streaming executor only.
- Attempt bodies, retry/pacing/`Retry-After`, epoch and monotonic clocks,
  response-size/runtime limits and safe public headers are typed and pinned.
  Every handled attempt returned to the collector retains bounded complete or
  partial evidence; an error cannot become valid empty history. Abrupt process
  death may leave an incomplete shard, never a positive admission.
- A strict-history-v2 store root is persistently bound for verified restart by
  `scope.json` to exactly one `HistoryRangeRequestV2`. A process-local plus OS
  file lock and owner-thread guard protect the pristine check through positive
  admission among cooperating writers; adversarial filesystem replacement/
  TOCTOU is outside this contract. Resume,
  repair, promotion of temp data and cross-request reuse are rejected.
- A manifest is evidence, not success. Positive admission is installed only
  after the complete raw/attempt/normalized/manifest graph is freshly reloaded
  and revalidated from disk within its budget. Untrusted temp hardlink aliases
  are reported but never read or double-counted.
- The S2v2→S3 adapter checks source-close timing in exact integer microseconds,
  binds the complete strict-history-v2 manifest and exact normalized row hashes,
  then projects to the frozen aggregation-v1 interface.
- Windows acceptance is atomic create-new/no-overwrite visibility plus
  process-crash/fresh-restart verification. It does not claim parent-directory
  or sudden-power-loss durability.

### Verification and operational receipt

```text
focused endpoint + strict v2: 83 passed, 1 skipped
all tests/v3:                174 passed, 1 skipped
frozen compatibility:        217 passed
full pytest:                  905 passed, 5 skipped,
                              2 known PytestCollectionWarning (21.80s)
code-scope red-teams:          APPROVE; P0/P1/P2 none
```

The `217` compatibility receipt is the exact pytest selection of
`test_strict_history_contract_v1.py`, `test_min1_aggregation_contract_v1.py`,
`test_frame_provenance_v1.py`, `test_closed_bar_contract_v2.py`,
`test_journaling_v2.py`, `test_scan_journal_reader_e2e_v2.py`,
`test_population_journal_v6.py`,
`test_population_journal_v5_compatibility_fixture.py`,
`test_population_journal_v2.py`, `test_population_journal_chain_v5.py`,
`test_journal_cycle_records_v2.py`, and
`test_mexc_strategy_v2_compatibility_fixture.py`.

No exchange request, public pilot, operational scanner/bot runtime, model,
Telegram, private API or capital action was run. `.env` was not opened.

### Remaining gate

U5 remains **not granted**. Per-shard limits are not aggregate acquisition
limits. Before any public request, the project still needs an immutable pilot
run manifest that pins the candidate endpoint identity, a bounded verification
procedure and expected receipt, symbols and exact ranges, one fresh artifact
root per shard, aggregate attempts/raw/storage/runtime and concurrency budgets,
disk preflight, stop conditions, detached result anchors and the intentionally
supplied executor. Full-universe orchestration and its global budget contract
remain future work. The Windows sudden-power-loss boundary must be accepted
explicitly or replaced by a stronger storage profile. After separate U5, the
first network action is the verification probe; mismatch STOPs before any
history acquisition.

## Bounded offline P2 QA-pilot run-contract checkpoint — 2026-08-16

This append-only checkpoint is the authoritative executable-state update after
the preceding strict-history-v2 checkpoint and supersedes its “Remaining gate.”
It completes the offline manifest/budget/pure-state contract for the bounded P2
QA pilot. It does not instantiate, authorize or execute a pilot and does not
claim full-universe acquisition, P3 admission or strategy edge.

### Publication and exact identity

```text
17b47c7 docs(roadmap): record pre-pilot history hardening
5595679 feat(data): add bounded MEXC QA pilot run contract
```

- The published code checkpoint is `5595679`, whose parent is `17b47c7`;
  later documentation-only descendants do not change its executable identity.
- `mexc_public_qa_pilot_run_v1` is pinned to
  `f3d642d436e9d4a44e65f35c6ea8375bd92b4b36b30f1c86af54936a608ce65e`.
- The implementation is `trading/market_data/mexc_pilot_run.py`; its focused
  contract tests are `tests/v3/test_mexc_pilot_run_contract_v1.py`.

### What is complete

- The offline contract canonically binds repository/docs/executor identities,
  exact QA symbols and ranges, deterministic ordered shards, one fresh store
  root per request, endpoint verification before acquisition, and hard global
  attempt/raw/storage/output/runtime/concurrency/spacing/inventory budgets.
- A manifest cannot grant U5. A detached authorization receipt is separately
  bound to the exact manifest, time window, domains, operations, caps, restart
  policy and storage-risk decision. Preflight freshness and the remaining
  worst-case run must fit the authorization window before each network stage.
- Pure state transitions require one durable network-intent slot per stage,
  fail closed on any pre-existing or unresolved slot, charge started attempts
  and terminal failures, and require publish/reload/detached-anchor evidence for
  endpoint, shard, terminal-failure and final positive outcomes. Candidate and
  sealed artifacts remain separately inventoried so their identities are
  acyclic and independently auditable.
- Positive completion is contractually gated on a freshly scanned/reloaded
  final full-inventory receipt and detached anchor. The pure projection checks
  their bindings and order but does not itself observe the disk. Partial or
  resumed runs are not positive admission.
- The endpoint fixture remains exactly `candidate_not_u5_verified`. Neither
  current official-document evidence nor live endpoint validity was established
  by this offline checkpoint.

### Verification and independent review

```text
focused pilot contract: 20 passed
all tests/v3:          194 passed, 1 skipped
full pytest:           925 passed, 5 skipped,
                       2 known PytestCollectionWarning
read-only audits:      2 x APPROVE; P0/P1/P2 none
```

No executor, concrete run-manifest instance or actual authorization was
created. U5 was not granted. No network request, public pilot, scanner/model
runtime, Telegram, private API or capital action was run, and `.env` was not
opened.

### Exact boundary and next gate

The module is a pure state/orchestration contract and deliberately contains no
HTTP/run executor; its narrow immutable-manifest publication/loading helpers do
not execute a run. A future intentionally supplied executor must actually
implement OS create-new intent arbitration and generate the fresh disk
scan/reload/detached-anchor evidence required by the contract. Its exact
version/hash must then be bound into a reviewed concrete pilot manifest; neither
the contract class nor a materialized state snapshot is that manifest or
authorization.

The filesystem threat model is
`cooperating_writers_plain_non_reparse_parent_chain_point_in_time_validation_v1`.
It does not prove safety against adversarial replacement/TOCTOU, and the Windows
profile does not prove sudden-power-loss durability. A later U5 receipt must
explicitly accept that Windows boundary or select a stronger storage profile.

Therefore the next permitted work remains offline: implement and independently
review the bounded executor, freeze its identity, instantiate and review the
exact QA-pilot manifest and its aggregate caps, and prepare the detached
authorization/verification procedures without creating an authorization
receipt. U5 remains a separate explicit user decision. Only after U5 may the
run acquire fresh official endpoint evidence and perform the bounded live
candidate probe; any
mismatch stops before history acquisition. Full-universe orchestration and its
P3 global policy remain later work, as do v3 model/scanner/threshold selection,
Telegram and any private/testnet/live execution scope.

## Official-evidence contract pin and contract-hash cache checkpoint — 2026-08-17

Branch `claude/codex-project-review-04581e`, parent `0032f1e`. Two independent
changes: freezing the previously unpinned official-evidence contract, and
removing a large amount of redundant hashing work from the contract layer.

### Official endpoint evidence contract is now pinned

`trading/market_data/mexc_endpoint_official_evidence.py` and its test arrived as
untracked work whose `_PINNED_CONTRACT_HASH` was still the empty string. The
guard that compares the computed digest against that pin is written as
`if _PINNED_CONTRACT_HASH and digest != _PINNED_CONTRACT_HASH`, so an empty pin
made it inert: the module was the only contract in the project shipping without
an active drift check, and its test asserted the empty value, locking that state
in. The pin is now
`421802f03282ea5f61f253607001036e80a1933e1d1ea16449c5ee261889e04d`
and the test asserts that literal plus agreement with the computed digest.
Mutating the declarative schema in memory now raises
`official_evidence_contract_changed_without_version_bump`, which it did not
before.

This changes nothing about the module's authority. Its only provenance mode
remains `reviewed_fake_fixture_only`; it still contains no HTTP client and no
default executor, and it still cannot authorize U5, a live probe, acquisition or
a terminal pilot receipt. Pinning freezes an offline contract; it does not
supply official endpoint evidence, and the network adapter that would produce
that evidence does not exist.

### Contract hashes are cached by value

The suite had grown from `22.90s` to `276.90s`. The cause was not test design
but recomputation in the contract layer: a single test issued `1,494,807` calls
to `_canonical_bytes` that produced `8` distinct results, and `2,235,008` calls
to `_path_parts` over `538` distinct inputs. Canonical JSON encoding, not
SHA-256, dominated the profile.

`strict_history._frozen_contract_hash` now memoizes contract hashes keyed by
value rather than by instance, which also collapses rebuilt-but-equal contracts.
`strict_history_v2`, `mexc_futures_transport`, `mexc_pilot_run`,
`mexc_pilot_output_layout`, `mexc_pilot_local_coordinator` and
`mexc_pilot_local_executor` route their `_sha256_payload(self.as_dict())`
identity properties through it; `mexc_pilot_output_layout._path_parts` is
`lru_cache`d because the pairwise overlap checks are quadratic.

`mexc_endpoint_official_evidence` is deliberately excluded. It canonicalizes
differently from every other module — `_canonical_json_bytes` appends a trailing
newline before hashing — so routing it through the shared helper silently
changes every hash it produces. That was attempted, caught by its pinned
fixtures as `24 failed`, and reverted. `tests/v3/test_frozen_contract_hash_cache_v1.py`
now states this divergence explicitly so the substitution is not retried.

### Validation

Full pytest `1229 passed, 10 skipped` in `62.00s`, with the same two known
collection warnings — `4.4x` faster than the `276.90s` parent, and `1225` of
those passes are the parent's exact set. The pinned hashes
`mexc_public_qa_pilot_run_v1` = `f3d642d4…e65f` and
`mexc_public_qa_pilot_output_layout_v1` = `cb19e6a5…e934` are unchanged, which
is the evidence that the cache is a pure accelerator.

No network request, public pilot, scanner/model runtime, Telegram, private API
or capital action was run, and `.env` was not opened. U5 remains ungranted and
the next acceptance gate is unchanged: implement and independently review the
bounded executor, then instantiate and review the exact QA-pilot manifest.
