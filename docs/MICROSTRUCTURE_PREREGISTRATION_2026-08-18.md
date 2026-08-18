# Microstructure experiment preregistration — 2026-08-18

Status: **declared before collection**. Nothing has been collected. This document
fixes the question, the outcome, the analysis and the failure criterion in
advance, so that a null result is accepted rather than re-screened until
something appears.

That failure mode is not hypothetical here. This project has already retracted a
`+2.07%` result that was a DCA artifact, and on 2026-08-18 an apparent model edge
survived three rounds of analysis before being traced to a distorted outcome
proxy. Both were caught because a control existed. This document exists so the
next one is caught before it is believed rather than after.

## The question

Does order book and trade tape state, observed at the moment a pump is fading,
carry information about whether price falls before it rises again?

That is the narrow question. It is not whether microstructure is useful in
general, and it is not whether the resulting model would be profitable to trade,
which is a separate question gated behind this one.

## Background this rests on

Measured on 2026-08-18 and recorded in `docs/AI_HANDOFF.md`:

- After a pump, the adverse and favourable excursions are symmetric. Median
  adverse `4.36%`, median favourable `4.27%`, ratio `0.981`, against `2.07%` and
  `1.99%` for matched random entries.
- 252 buckets over 34 price and volume features, and 205 buckets over 46 richer
  features including funding, levels, Fibonacci and divergence, produced nothing
  profitable on a time holdout.
- Eleven model configurations across two families ranked a holdout no better than
  chance; on corrected first-touch outcomes the best-ranked slice was the worst.

The hypothesis under test is therefore not "the strategy works". It is "the
information that would make it work is in a channel we have never recorded".

## Population

An episode is a triggered recording. Trigger: run-up `>= 10%` over 24 Min5 bars,
window high within the last 6 bars, at least a fifth of the run given back,
evaluated at bar close. Non-overlapping per symbol.

**Every triggered episode is paired with one control episode**, drawn at random
from the same universe and the same session, at an instant that did not trigger.
Analysis compares against these controls, never against zero.

**Episodes dropped for any reason are recorded as dropped**, with the reason.
A silently missing episode correlates with busy markets and would bias the sample
toward quiet ones.

Target size: 2000 triggered episodes with their controls. Below 1000 the analysis
is declared underpowered and reported as such rather than interpreted.

## Features

Derived from the depth and tape captured in the sixty seconds preceding the
decision bar close, and never from anything after it:

- book imbalance at several depths, and its rate of change
- total resting size on each side, and its rate of change
- spread, and its variability
- realised trade aggression: taker buy volume against taker sell volume
- trade size distribution, and the share of volume in the largest trades
- cancellation intensity, insofar as sequential snapshots reveal it
- depth consumed at the peak relative to the book before it

All are computed inside the formation window unconditionally, so their presence
never depends on the trigger.

## Primary outcome

First-touch net return of a short entered at the open of the bar following the
decision, with symmetric 3% stop and target, closed at the horizon otherwise,
charged `0.217%` round trip. This is the same definition used in
`ai/excursion.py`, whose ordering and causality guarantees are pinned by tests.

Secondary: the same at 5%, and the asymmetry rate.

## Analysis

1. Split by calendar time, earlier 70% for discovery, later 30% held out, with an
   embargo of one horizon.
2. Screen every feature on discovery, reporting all of them, not a top slice.
3. Fit a model on discovery over all features.
4. Apply both **once** to the holdout.
5. Bootstrap the surviving result clustered by symbol.
6. Repeat the whole procedure on shuffled labels to establish what it yields when
   nothing is there.

A permutation null alone is explicitly not sufficient. On 2026-08-18 a bucket
passed one at `p = 0.012` and then inverted out of sample, because shuffling
destroys the regime structure the bucket was fitting.

## Failure criterion, declared in advance

The hypothesis is rejected if, on the holdout:

- no single feature bucket is profitable, **and**
- the model's top decile is not profitable with a symbol-clustered bootstrap
  interval excluding zero, **and**
- the shuffled-label procedure produces a comparable result at `p >= 0.05`.

On rejection the result is recorded and the channel is closed. It is not
re-screened with different thresholds, different horizons, different episode
lengths or a different outcome definition. Any of those is a new preregistration
with its own collection.

If the holdout is positive, that is a candidate and not a conclusion: it requires
a second, later holdout collected after the result was stated.

## What this experiment cannot settle

It tests recorded microstructure. It does not test news, cross-exchange flow, or
the discretion of a human trader who filters candidates by judgement. A null here
does not mean no edge exists anywhere; it means this channel does not carry one
at this resolution on this population.

## Standing constraints

Public market data only. No credentials, no private endpoints, no Telegram, no
testnet, no live or paper execution. The scanner stays stopped. No trade is
placed, sized, or suggested on the basis of anything collected here, and no model
trained on it is enabled for any runtime.
