# Microstructure collection design — 2026-08-18

Status: **design only**. Nothing here is authorized, implemented or running. It
describes what would have to be collected, and under what guarantees, for the
open question to become answerable. U5 remains ungranted and the scanner remains
stopped.

## Why this exists

Every screen run to date failed for the same reason, and it is not model
capacity. On 2026-08-18 a gradient booster was given 46 features over 17893
events — funding, support levels, Fibonacci distances, RSI divergence,
liquidation distances, BTC return, relative strength — and ranked a time holdout
no better than chance (`p = 0.335`). Before that, 252 buckets over 34
price-and-volume features produced nothing that survived out of sample.

The channels a human actually reads when deciding whether a pump is exhausted are
absent from every one of those screens:

| channel | present? |
|---|---|
| OHLCV klines | yes, Min5/Min15/Min60/Hour4 |
| funding rate | partially, in `pump_dataset_full` only |
| order book depth | **no** |
| trade tape | **no** |
| open interest | **no**, placeholder zeros only |
| timestamped news | **no**, channel designed but empty |

`core/market_data.py:168` can request depth, but it targets the Bybit
`/v5/market/orderbook` endpoint, it is a live call rather than stored history,
and no MEXC kline API serves depth or tape retrospectively. This is therefore a
collection problem: the data does not exist anywhere and has to be recorded
going forward.

## What breadth actually buys

The reason to automate is parallel coverage. A person watches a handful of
symbols; a collector watches the whole universe and never blinks. That advantage
is real and it is not contradicted by the null results above — those say only
that breadth over the *current* inputs finds nothing, because there is nothing in
them to find.

It does not follow that microstructure will contain an edge. It follows that the
question is currently untestable, and this design makes it testable.

## Tiers

Rate budget is the binding constraint. `trading/market_data/mexc_client.py:111`
sets 8 requests per second, and depth plus tape for one symbol at one-second
resolution consumes two of them.

**Tier 0 — universe, always on.** One kline sweep of every tradable symbol every
five minutes, plus the bulk ticker, plus funding at its own cadence. At roughly
500 symbols this is about 1.7 requests per second sustained and establishes the
denominator: every symbol, every interval, whether or not anything happened.

**Tier 1 — warm watchlist, rolling buffer.** Symbols showing early elevated
activity get a depth snapshot every ten seconds into a rolling in-memory buffer
of the last thirty minutes. This exists because the interesting question is what
the book looked like *before and during* the peak, not after it. A trigger that
starts recording at the moment it fires has already missed the evidence.

**Tier 2 — hot capture.** When a candidate fires, its buffer is flushed to disk
and depth plus tape are captured at one-second resolution for a preregistered
duration. Two requests per second per active episode, so with Tier 0 running the
sustainable concurrency is about three episodes.

## The control requirement

This is the part most likely to be dropped and the part that decides whether the
resulting data is analysable at all.

**Tier 2 must also fire on randomly chosen symbols and instants that did not
trigger.** Without them the collected population contains only events, so no
comparison against a matched non-event is possible, and the dataset reproduces
exactly the gate-conditioned missingness this project already carries in
`level_dist` (30.6% missing, correlated with the gates). Every screen in this
repository that produced a decisive answer did so because a matched random
control existed. One control episode per triggered episode, drawn from the same
universe and the same session.

Equally: when concurrency is exhausted and an episode is dropped, **the drop must
be recorded as an explicit outcome**. A silently skipped episode is a third form
of conditioned missingness, and it will correlate with exactly the busy market
conditions that matter most.

## Budget

Per episode, thirty minutes at one-second resolution: roughly 1800 depth
snapshots at about 400 bytes compressed, plus a few thousand tape entries, so
under one megabyte. At twenty episodes a day — ten triggered, ten control — that
is about 16 MB a day, or half a gigabyte a month. Storage is not the constraint;
request rate and calendar time are.

## Time to an answerable question

The screens that produced today's conclusions used between 8000 and 18000 events.
At ten triggered episodes a day a comparable population takes well over a year.
Reaching a few thousand events inside a quarter requires either a looser trigger
or a wider universe, and that is a scope decision to take before collection
starts rather than after.

State this plainly in the preregistration: a collection that yields 300 events is
not a smaller version of this analysis, it is an underpowered one that will
produce a confident-looking bucket and no way to tell whether it is real.

## Gates that precede any of this

1. **U5** — an explicit, detached authorization from the user. Not granted.
2. **Official endpoint evidence** for depth and tape. Neither path is known to
   this project: only `contract/kline`, `contract/ticker`,
   `contract/funding_rate` and `contract/detail` appear anywhere in the code.
   `trading/market_data/mexc_endpoint_official_evidence.py` is the contract that
   exists for exactly this, and its current provenance mode is
   `reviewed_fake_fixture_only`.
3. **A bounded live verification probe** against those endpoints, succeeding,
   before any acquisition request.
4. **A concrete run manifest** with its aggregate caps, reviewed.

Steps 2 through 4 are the pilot machinery Codex already built. This design gives
that machinery a purpose it did not previously have: it was assembled to acquire
Min1 klines for a strategy since shown to be symmetric, and it is better spent
acquiring the channels that were never measured.

## What this is not

Not a signal, not a strategy and not a claim that microstructure contains an
edge. It is the instrumentation required to ask. The failure criterion must be
declared before collection begins, so that a null result is accepted rather than
re-screened until something appears — which is the failure mode this project has
already recorded twice.
