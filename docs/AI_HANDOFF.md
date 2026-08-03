# AI collaboration handoff

> [!IMPORTANT]
> **Historical audit log — not the current single source of truth.**
> Read `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md` first. This file mixes reviews from
> different points in time and from two independent, unmerged worktrees:
> root/Bybit and Claude/MEXC. Earlier positive DCA, expectancy, portfolio, and
> safe-leverage conclusions were superseded by the local MEXC commit `9f71a86`,
> whose executable single-position replay found no demonstrated edge after
> costs. Those preservation commits and the later causal scanner/contract series
> are now published; statements below that the MEXC branch is nine commits ahead
> of GitHub are retained only as dated history.
> Never use this file alone to choose a branch, tune the strategy, or authorize
> trading.

Current checkpoint: 2026-08-03, Europe/Moscow

- Root functional anchor: `e501e21`, published; validation
  `532 passed, 4 skipped, 3 warnings`.
- MEXC functional anchor: `98217df`, published; validation
  `340 passed, 4 skipped, 2 warnings`.
- MEXC is the selected target exchange. Its scanner is causal,
  population-journaled, public-data and signals-only. The executable research
  contract is one SHORT entry, one stop, one TP and concurrency one.
- The no-edge verdict remains in force. The next model is a shadow LightGBM on
  versioned single-position runtime-population labels; causal TCN/Chronos are
  challengers and LLMs are context-only.
- Credentials were not rotated and old Git history was not cleaned. Current tips
  are sanitised, but private API, Telegram, testnet and live remain blocked.

The detailed MEXC runtime checkpoint is also recorded in that worktree's own
`docs/AI_HANDOFF.md`.

## Independent review of `claude/codex-project-review-04581e`

### Scope and collaboration decision

- The user selected **MEXC as the target exchange** and the Claude branch as the
  implementation base. Codex is acting as an independent reviewer.
- Reviewed worktree/commit:
  `claude/codex-project-review-04581e` at
  `68e0ff77373db2b48c19e3dcf09f1cbb7d569e47`.
- No merge, strategy change, threshold change, source-code change, or runtime
  start was made during this review.
- `data/processed/pump_dataset_v2.csv` contains 5,042 events across 88 symbols.
  MEXC timestamps a kline with its **open** time, while the 1h signal is only
  knowable at the close. The corrected label window is therefore
  `[event.ts + 3600, event.ts + 3600 + 48h]`.
- The Claude worktree now contains that `decision_ts = event.ts + 3600` fix in
  `ai/pump_dataset.py:285-304`; the comparisons below quantify the damage in the
  previously built CSV rather than modifying it.

### A. Look-ahead damage

Primary requested check:

- 25 symbols were selected deterministically across five event-count strata.
- 1,378 events had complete paired 5m windows; 11 boundary/cache-gap events were
  excluded.
- The old window was reconstructed from cache and reproduced every stored
  `mae_pct`, `mfe_pct`, `dca_resolved`, and `n_averages` value (floating-point
  error below `1e-15` and zero categorical/count mismatches).
- Confidence intervals below use 50,000 symbol-cluster bootstrap draws, so
  repeated events from one symbol are not treated as independent symbols.

| Metric | Old window from `event.ts` | Corrected `+1h` window | Change |
|---|---:|---:|---:|
| Mean `mae_pct` | 15.8689% | 15.9153% | +0.0464 percentage point |
| Median `mae_pct` | 6.6329% | 6.5936% | -0.0393 pp |
| Mean `mfe_pct` | 10.5547% | 10.1989% | -0.3558 pp |
| Median `mfe_pct` | 7.6400% | 7.4790% | -0.1610 pp |
| `dca_resolved` | 91.3643% | 86.7925% | **-4.5718 pp** |
| Mean `n_averages` | 0.2409 | 0.4419 | **+83.43%** |

The 95% symbol-cluster CI for the MAE change is
`-0.0425..+0.1106 pp`; MAE is therefore essentially unchanged. The CI for the
resolved-rate change is `-6.5028..-3.1295 pp`, and the CI for the mean-add
change is `+0.1257..+0.2630`, so the DCA changes are not sampling noise.

Transitions and direct leakage attribution:

- 65 events changed `resolved -> unresolved`; only 2 changed in the other
  direction.
- `n_averages` increased for 161 events, decreased for 18, and was unchanged
  for 1,199.
- 664/1,378 events (48.19%) had already touched the 3% target inside the signal
  hour, before the signal could exist.
- All 65 lost resolutions and 157/161 increased-add cases came from that
  contaminated hour. In addition, 95.50% of events had a signal-hour high above
  the eventual hourly close/entry.
- MAE class rates hardly moved (`<=3%`: 26.78% -> 27.65%; `<=5%`:
  40.93% -> 41.15%; `<=8%`: unchanged at 54.57%) because later 48h extrema
  normally dominate the global maximum.

A separate full-dataset sensitivity pass over all 5,042 events confirmed the
same direction and magnitude:

| Metric | Old all-event value | Corrected all-event value | Change |
|---|---:|---:|---:|
| Mean `mae_pct` | 17.6074% | 17.6656% | +0.0582 pp |
| Mean `mfe_pct` | 11.9100% | 11.5908% | -0.3192 pp |
| `dca_resolved` | 92.4831% | 88.6355% | **-3.8477 pp** |
| Mean `n_averages` | 0.2376 | 0.4351 | **+83.14%** |

For the previously reported final filter
(`atr_pct_1h > 0.0471497`, `move_pct > 0.0841911`,
`relative_strength > 0.0530588`, `rsi_4h > 61.6422`, 240 events):

| Result | Old labels | Corrected labels |
|---|---:|---:|
| Resolved | 100.000% | 98.333% |
| Mean adds | 0.254 | 0.692 |
| Worst code `peak_dd` | 1.260 | 5.684 |
| Claude score with `T=0.05` | +0.06271 | +0.01606 |
| 3%-consistent score with peak-DD loss | +0.03861 | **-0.01251** |
| Force-close-at-48h score, no costs | not evaluated | +0.01905 |

Verdict: the timestamp bug is cosmetic only for aggregate MAE and modest for
aggregate MFE. It is **material** for `dca_resolved`, number/depth of averaging,
worst drawdown, and every expectancy claim derived from those fields. The old
claims of 100% recovery and shallow DCA do not survive the correction.

### B. Min1 versus Min60 mismatch

There were no cached MEXC `Min1` files, so the exact requested check used fresh
public MEXC completed candles on 2026-07-26; it used no credentials and did not
start the bot. Ten symbols were checked at 33 aligned hourly decision points
each (330 paired observations):

`1000BONKUSDT`, `BANKUSDT`, `CHILLGUYUSDT`, `DEXEUSDT`, `DOGSUSDT`,
`FILECOINUSDT`, `MORPHOUSDT`, `PEPEUSDT`, `TRUMPOFFICIALUSDT`, and
`ZROUSDT`.

The calculation used the production indicator code for `atr / close` and the
production gate's exact `sum(close * volume)` over 12 bars.

| Comparison | Min60 / Min1 |
|---|---:|
| Median ATR-percent ratio | **10.34x** |
| ATR-percent ratio p10 / p90 | 6.25x / 18.56x |
| Median 12-bar volume ratio | **89.18x** |
| 12-bar volume ratio p10 / p90 | 26.65x / 471.07x |

| Threshold | Min1 pass rate | Min60 pass rate |
|---|---:|---:|
| `atr_pct >= 0.046` | **0/330 (0.00%)** | 59/330 (17.88%) |
| `sum(close*volume, 12) >= 100000` | 76/330 (23.03%) | 198/330 (60.00%) |

Per-symbol median ATR ratios ranged from 7.75x (`TRUMPOFFICIAL`) to 15.90x
(`MORPHO`). Per-symbol median volume ratios ranged from 61.87x (`PEPE`) to
290.41x (`MORPHO`).

Two interpretation cautions make the mismatch more serious, not less:

1. Twelve Min1 bars cover 12 minutes, while twelve Min60 bars cover 12 hours.
   The original volume calibration actually summed twelve Min5 bars (one hour);
   applying the same number to twelve Min1 bars changes the horizon again.
2. `MexcContractClient` maps API `vol` directly and the gate multiplies it by
   price without applying contract size. Across contracts this value is not
   guaranteed to be true USDT notional despite the name
   `min_hourly_usd_volume`.

A cached 10-symbol Min5/Min60 lower-bound check over 21,807 aligned historical
points already showed a 3.99x median ATR ratio, a 13.42x median volume ratio,
ATR pass rates of 0.39% versus 8.76%, and volume pass rates of 40.01% versus
73.95%.

Verdict: the 1h thresholds are not portable to a 1m runtime. At Min1 the ATR
gate was absolute in this sample, and the volume gate measures a different
time horizon and possibly different units.

### C. Expectancy audit

There is no committed expectancy function. Claude's scratchpad/session scripts
used:

```text
score = target * (n_averages + 1)          if dca_resolved
        -dca_peak_drawdown_units           otherwise
```

#### Critical target/label mismatch

`LabelConfig.dca_target_pct` is 3% (`ai/pump_dataset.py:43-49`), and
`ai/build_pump_dataset.py` does not override it. Later headline calculations
substituted `target=0.05` into the reward branch without rebuilding
`dca_resolved`, `n_averages`, or peak drawdown for a 5% exit. With fixed labels,
raising `target` makes the score rise by construction.

An independent full replay of all 5,042 old-window events at each actual target
shows the sign error:

| Actual target/replay | Actual resolved | Correct replay score | Claude score on fixed 3% labels |
|---|---:|---:|---:|
| 2% | 97.660% | +0.011148 | -0.011212 |
| 3% | 92.483% | -0.000391 | -0.000391 |
| 5% | 78.144% | **-0.031719** | **+0.021251** |

On the chronological test of the final rule, the reported 5% score changes from
`+0.059524` to `-0.044530` when the labels are actually replayed at 5%; worst
drawdown is 8.4056 units. This alone invalidates the published 5%-expectancy
conclusion, even before removing look-ahead.

#### Unresolved and averaging bias

`dca_peak_drawdown_units` is the worst historical floating loss, not a realized
exit:

```text
(bar_high - arithmetic_average_entry) / arithmetic_average_entry * legs
```

For the 379 unresolved old-window events, peak loss had mean 0.4371 and median
0.0836 units, whereas terminal 48h MTM loss had mean 0.2190 and median 0.0367;
16.9% were in a small floating profit at the horizon. Replacing the impossible
"realize the future worst point" convention with a no-cost 48h force close
changes the all-event 3% score from `-0.000391` to `+0.016002`. Thus the
unresolved penalty alone is pessimistic, but it is not executable.

The systematic pro-averaging asymmetry is elsewhere:

- Resolved trades receive `+target * legs` regardless of the drawdown they had
  to survive. Of 4,663 old-window winners, 59 had peak DD above 0.5 initial-leg
  units, 29 above 1, 8 above 2, and 3 above 5; the maximum was 11.843.
- A stop/liquidation rule would have to be replayed pathwise for **all** events.
  Applying worst DD only to the unresolved tail creates a binary cliff between
  an eventual target touch and a near miss.
- Extra legs are rewarded without dividing by deployed capital or accounting
  for occupied capital/concurrency. Fees, spread, slippage, funding,
  liquidation, margin mode, cash limits, and opportunity cost are absent.
- Code DD is normalized by the current arithmetic average rather than initial
  entry; in initial-entry units it is understated by the average/entry ratio,
  up to 24% with six 8% adds.
- The arithmetic blend is valid only for equal-quantity legs. If the intended
  legs have equal USDT size, the blend must be harmonic; the current replay is
  then too easy. An equal-dollar replay produced 3% score `-0.002363` and 5%
  score `-0.039745`.
- The per-bar replay adds at the high and then permits a take-profit at the low
  of the same 5m bar. This affected only about 0.64% of winners and changed the
  native 3% score by only `-0.000083`; it is real but small compared with the
  one-hour leakage and target mismatch.

Also, `mae_pct` and `mfe_pct` are extrema over the complete 48h label horizon,
not extrema of the position truncated at its resolution. They must not be
reported as realized trade excursions.

Verdict: this score can be a rough recovery/path diagnostic at one fixed target.
It is not expected PnL and cannot establish profitability or DCA safety. Its
unresolved convention is pessimistic, while its resolved/capital treatment is
optimistic toward deep eventual recovery; the net sign depends on an arbitrary
exit convention, which the corrected final-rule example demonstrates
(`-0.01251` peak-loss score versus `+0.01905` no-cost terminal-close score).

### D. Measured or implemented but not wired to runtime

The exact current decision path is:
`app/main.py:165-177` fetch/build -> `app/main.py:179-192` strategy ->
`app/main.py:194-211` risk/execution -> `app/main.py:247-251` alert.

| Component | Actual state | Required `app/main.py` connection point |
|---|---|---|
| MEXC public market data | **Not wired.** `MexcContractClient` exists, but runtime constructs `MarketDataFeed(base_url="https://api.bybit.com")` and that feed constructs the Bybit client internally. | Replace/refactor feed construction at `:279-290`; make the client injectable. Then the existing fetch at `:165` and HTF cache at `:313-316` share the MEXC backend. |
| MEXC private execution/reconciliation | **Not implemented/wired.** Runtime uses `BybitAdapter` at `:286`; the MEXC client is public-data only. | A private adapter is required before replacing `:286`. It must satisfy reconciliation and execution consumers at `:101`, `:118-126`, `:194-211`, `:290-310`, and startup reconciliation at `:332-337`. |
| Dynamic MEXC universe | **Not wired.** `SymbolUniverse` is used only by offline dataset building/tests; runtime receives static `BOT_SYMBOLS` from `bootstrap.py:124`. | Construct before `:286-287`; refresh before startup reconciliation at `:332`, then at the loop boundary `:363` (or cycle start `:101`); replace `cfg.symbols` at `:333`, `:365`, and loop input `:116`; update WS membership at `:287`. |
| Alternative.me sentiment | **Already wired.** | Constructed at `:319-330`, fetched once per cycle at `:103-106`, placed in extras at `:171-173` and context at `:187-188`. No missing runtime hook. |
| Funding, open interest, long/short ratio | **Not wired.** Funding and long/short are hardcoded `None` at `:174-175`; OI is absent. | Populate from the MEXC cycle snapshot immediately before feature build at `:171-177`, then extend `StrategyContext` at `:180-192` for OI/derived fields. |
| `SignalObservationTracker` | **Not wired in this branch.** | Instantiate near `RuntimeStore`/strategy at `:296-318`; pass through `run_cycle` (`:83-100`, `:364-380`); `update_frame` after `:165-170`; record after intent/alert decision at `:180-192`/`:247-251`; expire stale observations at `:382-385`. |
| `SignalPositionTracker` | **Not wired in this branch.** | Instantiate/pass at the same points; update marks after `:179`; record only delivered/accepted shorts at `:247-251`; close on exits around `:211-214`. Manual-versus-auto semantics must be explicit. |
| Signal chart and liquidation-density panel | **Not wired.** `alerts/chart_generator.py` has builders and the legacy Telegram client can send photos, but there is no call site in runtime. | Build after signal generation (`:177-192`) and send with the accepted alert (`:247-251`); the active alert abstraction must expose photo sending. Populate liquidation inputs before feature build. |
| HTF cache | **Syntactically wired but functionally fail-open.** It is constructed at `:313-316`; its default `Hour4` is a MEXC token, but the current backend at `:289` is Bybit. Empty HTF data silently skips the gate. | Fix backend at `:289`; add explicit data-quality status before strategy call at `:180`. Do not treat an empty cache as a passed filter. |
| `min_rsi_1h` and `require_confluence` | **Declared but never consumed.** RSI15, Fibonacci, divergence, confluence, level and liquidation features are dataset/chart-only. | Build a closed-bar multiframe context after pipeline output at `:177` and before `strategy.generate` at `:180`; pass it explicitly to the strategy. |
| Cross-sectional volatility | **Wired internally**, but inherits the current-timeframe problem. | Rebase its inputs on calibrated Min60 data at the feed/context boundary `:165-180`. |
| BTC relative strength | **Wired but fail-open** and uses `cfg.timeframe` at `:112`, currently Min1 rather than calibrated Min60. | Fetch/validate the benchmark on the calibrated timeframe at `:108-114`; propagate data-quality failure before `:180`. |
| Overhead-level gate | Implemented but disabled by default and lacks runtime env/config mapping. | Add explicit config when constructing strategy at `:311-312`; provide its closed-bar level context before `:180`. |
| `max_chase_atr` | Implemented, but default `0` disables it and there is no runtime override. | Map an explicit reviewed config at strategy construction `:311-312`; do not activate from the biased calibration. |
| Other measured pump features | Exhaustion, wick, acceleration, idiosyncratic return, EMA50, consecutive-up bars, and several structural fields remain dataset-only. | Select only validated fields, compute in `FeaturePipeline` at `:177`, and pass through `StrategyContext` at `:180-192`. |
| ML inference | Flags/model support exist, and `FeaturePipeline` output is built at `:177`, but no inference result is passed to the strategy. | Add shadow inference between `:177` and `:180`; keep it non-blocking until separately validated. |
| DCA plan, `partial_tps`, `max_safe_leverage` | Measured/calculated, but runtime emits one intent and does not execute the offline DCA/partial-exit model. | Reporting can attach to `:247-251`; real execution would require an explicit, tested state machine across `:194-211`, not a strategy-threshold patch. |

History collection, dataset construction, and label generation are intentionally
offline and should not themselves be inserted into `run_cycle`. What runtime
needs is a MEXC feed/universe plus a small, closed-bar, validated feature
contract derived from that research.

### Review conclusion

1. Rebuild `pump_dataset_v2.csv` after the `decision_ts` fix before accepting
   any calibrated threshold or recovery statistic.
2. Re-run every target at its actual target; do not change only the positive
   reward constant.
3. Define executable PnL first: sizing basis, capital denominator, terminal
   exit/stop, liquidation/margin mode, fees, slippage, funding, and concurrency.
4. Calibrate and run on the same timeframe and volume horizon/units.
5. Wire and validate MEXC public data/universe first. MEXC private execution is
   a separate safety-critical adapter task.
6. Keep the bot stopped; this review provides no basis for live trading.

## Prior root-worktree state (2026-07-25; retained for context)

- Branch: `feat/phase2-layer1-pump-runtime-alignment`.
- The worktree contains many intentional tracked and untracked changes. Inspect
  the diff before editing and do not discard unrelated work.
- The bot is intentionally **stopped** at the user's request.
- The Codex heartbeat automation `koteika` was deleted.
- Runtime remained `paper` / `dry_run=True`; live trading was not enabled.
- Last full test run: `529 passed, 4 skipped, 2 collection warnings`.
- SQLite quick check: `ok`; open inflight intents: `0`; duplicate order IDs and
  order-link IDs: `0`.

## Prior root-worktree Codex change

Problem: signal observations only completed when `update_frame()` was called for
their symbol. If a symbol left the discovery universe, an expired observation
remained active forever and could distort monitoring.

Implementation:

- `trading/state/signal_observation_tracker.py`
  - Added `expire_stale()`.
  - Wall-clock-expired records without a full data horizon are persisted as
    `expired_incomplete`.
  - Partial records include coverage metadata and are not treated as completed
    calibration samples.
- `app/main.py`
  - Calls `expire_stale()` after each full decision cycle.
  - Logs `signal_observation_expired_incomplete`.
- `tests/v2/test_signal_observation_tracker_v2.py`
  - Added regression coverage for a symbol leaving the runtime universe.

Validation:

- Targeted observation tests: `14 passed`.
- Full suite: `529 passed, 4 skipped`.
- Two historical stale rows were migrated to `expired_incomplete`; two recent
  rows remained active when the bot was stopped.

## Other material functionality already present in this worktree

- PyCharm root launcher and single-runtime file locking.
- Parallel full-universe analysis with cycle progress logging.
- Signal observation tracking without look-ahead.
- Shadow tracking of delivered manual short signals.
- Managed-exit alerts and intrabar TP/SL barrier handling.
- Conservative same-bar rule: stop loss wins a TP/SL tie.
- Early lifecycle cleanup after a managed or barrier exit to prevent duplicate
  resolution alerts.
- Telegram proxy detection and redaction of tokens from logs.

Relevant tests include:

- `tests/v2/test_early_lifecycle_v2.py`
- `tests/v2/test_market_parallelism_v2.py`
- `tests/v2/test_runtime_instance_lock_v2.py`
- `tests/v2/test_signal_observation_tracker_v2.py`
- `tests/v2/test_signal_position_tracker_v2.py`

## Important quality findings

- The latest restart discovered 348 eligible instruments and processed them
  with 16 workers. This is a filtered universe, not every Bybit contract:
  liquidity, listing age, contract quality, and data-quality gates reduce it to
  roughly 297–305 decision-ready symbols per cycle.
- Two newly delivered early `WATCH` signals, `EULUSDT` and `LIGHTUSDT`, crossed
  their stops almost immediately. Early/WATCH quality is therefore not ready
  for live-money entry.
- Earlier examples showed mixed outcomes: AKE benefited from a managed
  profit-protection exit, while several WATCH signals hit stops.
- Do not loosen thresholds merely to increase signal count. First separate
  informational WATCH alerts from actionable entries and study false positives
  using completed, no-look-ahead observations.

## Recommended next discussion

Before further strategy changes, align with the user on:

1. whether `WATCH` is informational or an immediate entry instruction;
2. the exact definition of a successful trade and evaluation horizon;
3. whether tokenized equities/TradFi contracts belong in the universe;
4. acceptable signal frequency versus maximum adverse excursion;
5. explicit monetary risk limits required before any live validation.

For calibration comparison JSON, follow `AGENTS.md` and use
`scripts/observation/triage_calibration_result.ps1`.

## Claude-worktree review checkpoint — 2026-07-26 05:16 MSK

The detailed independent review remains in
`.claude/worktrees/codex-project-review-04581e/docs/AI_HANDOFF.md`.
Latest material finding: none of the repository's chronological split paths
purges the 48-hour forward-label horizon. On `pump_dataset_v3.csv`, the
hypothetical 70/15/15 boundaries leave 130 train labels crossing into
validation and 97 validation labels crossing into test; the 80/20 path leaves
124 train labels crossing into test. The 70% positional cut also divides one
identical `decision_ts` cohort between train and validation.

No committed consumer or threshold-search script for pump v2/v3 was found, so
the threshold comments' claimed held-out results are not reproducible from the
repository. Group by decision time and add a full 48-hour purge/embargo before
any recalibration. Claude HEAD remains `8cc31fc`, unmerged and unpushed; the bot
remains stopped.

## Claude-worktree review checkpoint — 2026-07-26 05:46 MSK

Additional blocking research-validity defects are documented in the detailed
Claude-worktree handoff:

- The historical 100-day sample is selected by one current MEXC universe
  snapshot (current turnover, survival and 24h ranking), so early events use up
  to roughly 97 days of future selection information. No point-in-time universe
  or rejected-symbol manifest exists.
- V2 and v3 share only 67 event-bearing symbols despite being written about
  2.5 hours apart (Jaccard 0.609). V1/v3 Jaccard is 0.171.
- At least 303 v3 rows have insufficient history for a true 7d feature and
  1,516 for a true 30d feature; the code silently writes shorter-window values.
- Obvious equity/ETF proxies contribute 301 v3 rows (5.81%) and resolve at
  74.4% versus 89.0% overall, mixing a different market regime.
- Exact ATR/liquidity/4h-RSI thresholds and their OOS claims remain
  unreproducible. In particular, 0.046 ranks around the 75th–77th percentile,
  not the stated 80th, in all three pump artifacts.

No strategy/runtime files were changed by Codex. Claude HEAD remains
`8cc31fc`; no merge/push occurred and the bot remains stopped.

## Claude-worktree review checkpoint — 2026-07-26 06:16 MSK

Official MEXC documentation resolves the volume-unit question: kline `vol` is
a contract count and exact quote turnover is kline `amount`. The current MEXC
client discards `amount`, while the strategy computes `close*vol` and labels it
USD. In MEXC's BTC example this overstates turnover by about 9,998x because
`contractSize=0.0001`. Across cached 12-hour windows for 168 symbols,
contract-size adjustment changes the `$100k` gate from 58.5% passing to 82.3%,
with 11,841 raw false passes and 98,382 false fails. Existing caches cannot be
repaired exactly because `amount` was not persisted.

The label family is also internally mixed. In v3, 324 rows are
`dca_resolved=1` although the original-entry target used by
`time_to_target_min` never occurs; another 494 reach that original target only
after the modeled blended-DCA exit. Intrabar high-first ordering changes 25 v3
outcomes, including six resolution flags.

Detailed evidence and official links are in the Claude-worktree handoff.
No source changes were made; Claude HEAD is unchanged and the bot is stopped.

## Joint night review complete — 2026-07-26 06:46 MSK

The independent safe review is complete and the night automation can stop.
Claude HEAD remains `8cc31fc`, unmerged and unpushed. Final focused validation
is `39 passed`; the last stable full Claude-worktree run is
`243 passed, 4 skipped`. No trading/Python process is running, and live mode
was never enabled.

The next implementation should fix data provenance and contracts before any
strategy/threshold work: exact MEXC quote turnover, typed fetch failures,
point-in-time universe selection, full feature warm-up/cadence checks,
consistent trade-path versus fixed-horizon labels, and purged reproducible
calibration. Then validate a public-data-only MEXC scan path with execution
disabled. The detailed ordered checklist is in the Claude-worktree handoff.
