# Design decisions, trade-offs, and the interview-style walkthrough

This document is the "why", not the "what" — the README covers what each
piece does; this covers why it's built this way, what was tried and
rejected, and what an interviewer would probably push on.

## Table of contents

- [The core problem this system solves](#the-core-problem-this-system-solves)
- [Avoiding leakage](#avoiding-leakage)
- [Why the synthetic data isn't a straightforward simulation](#why-the-synthetic-data-isnt-a-straightforward-simulation)
- [Why Flink isn't on the synchronous decision path](#why-flink-isnt-on-the-synchronous-decision-path)
- [Why two feature systems (Redis + Feast)](#why-two-feature-systems-redis--feast)
- [The chronological split, and why it matters more than it sounds](#the-chronological-split-and-why-it-matters-more-than-it-sounds)
- [The decision engine's reason codes](#the-decision-engines-reason-codes)
- [What broke during development, and what that says about the design](#what-broke-during-development-and-what-that-says-about-the-design)
- [What was actually verified live, and what wasn't](#what-was-actually-verified-live-and-what-wasnt)
- [What I'd do differently at real scale](#what-id-do-differently-at-real-scale)

## The core problem this system solves

Fraud detection is a real-time behavioural anomaly problem, not a static
classification one. The same $4,500 transaction is unremarkable for a
high-value shopper and alarming for someone who has never spent more than
$200 — so the model's inputs have to describe *that user's own recent
history*, not just the transaction in isolation. That history changes with
every event, which is what makes this a streaming systems problem as much
as an ML one: the feature store has to be kept current in real time, not
recomputed once a day.

## Avoiding leakage

This was the first thing the project brief called out, and it shaped
everything downstream. The original skeleton this was built from had a
`ground_truth.py` that computed a "suspicion score" from a `BehavioralSignals`
struct, and a `historical_generator.py` that assigned each *event* (not
each *user*) a random profile category, then derived the label directly
from that category's hand-coded signal weights. Two problems with that,
independent of each other:

1. **The label wasn't a label — it was the features, restated.** A model
   trained on "does this row's category imply high suspicion" doesn't learn
   fraud detection; it learns to invert a lookup table the generator itself
   used to create the label.
2. **There was no persistent user identity**, so no temporal feature was
   even *computable*. A rolling "transactions in the last hour" is
   meaningless if the same `user_id` gets a different behavioural profile
   on every single event.

The fix (`src/data/user_population.py`, `src/producers/fraud_injection.py`,
`src/features/temporal_features.py`) separates three things that were
previously tangled together:

- **A user's persistent baseline** (their archetype, spending level, usual
  devices, active hours) — never touched by anything downstream as a
  feature.
- **The label**: whether a given transaction is part of an injected fraud
  episode. This is simulation ground truth, exactly analogous to a real
  system's confirmed-chargeback labels — known because we did the
  injecting, not derived from anything the model can see.
- **The features**: computed in `compute_features()` purely from a user's
  own event stream, using only events *strictly before* the one being
  scored (`closed` on the left edge of every rolling window). Two tests
  make this a checked property, not a claim:
  - `test_features_are_causal_wrt_future_events` computes features twice —
    once over the full event stream, once over a truncated prefix — and
    asserts every row that exists in both computations gets identical
    feature values. If a feature ever depended on something that hadn't
    happened yet, this test would catch the discrepancy directly.
  - `test_online_matches_offline_feature_computation`
    (`tests/test_feature_store_client.py`) replays the same event stream
    through the Redis-backed online client, one event at a time — read
    state, score, write state, exactly as production would — and checks
    every value against the offline batch computation. This is really a
    leakage test wearing a parity-test's clothes: the online client
    *cannot* see the future even if it wanted to, since events arrive one
    at a time: for the two computations to agree, the offline one can't be
    smuggling in anything the online one couldn't have known either.

## Why the synthetic data isn't a straightforward simulation

Two rounds of iteration were needed after the first version of the fraud
injection, both driven by the same failure mode: **the model was too good**.

The first working version scored **ROC-AUC 1.0000** on the held-out test
set. That is not a result to be happy about — it means the classes were
perfectly separable, which in a synthetic dataset almost always means the
injection process is too rigid rather than that fraud is actually this
easy. Inspecting feature importances confirmed it:
`seconds_since_last_event` alone carried over half the model's weight,
because every injected episode packed several events into a few seconds,
and *no benign user ever did that*. The model wasn't learning fraud
detection; it was learning "did the generator's fraud-injection code run".

The fix was to inject **noise that overlaps the fraud signal on purpose**:

- Benign users occasionally get a genuinely new device (`EventGenerator`'s
  `BENIGN_NEW_DEVICE_RATE`), an unusually large one-off purchase
  (`BENIGN_LARGE_PURCHASE_RATE`), a typo'd password (a short benign
  failed-login burst in `HistoricalDataGenerator._normal_timeline`), or —
  critically, since this was the feature that had been trivially
  separating everything — a rapid-fire burst of 2-4 legitimate
  transactions in a few minutes, simulating a real shopping session.
- Fraud episodes themselves got softer and more varied: `account_takeover`
  now sometimes has *zero* failed logins (a phished password can work on
  the first try), a quarter of the time reuses a device the user has
  genuinely used before rather than always inventing a brand-new one, and
  its purchase-amount multiplier is drawn from a wide range (1.3x-6x) that
  overlaps with an ordinary large purchase at the low end.

The result after these changes: ROC-AUC ~0.9998, but now with actual
mistakes in the confusion matrix (a handful of false positives and false
negatives at the chosen threshold) and feature importance spread across
eight-plus features rather than concentrated in one. That's a result worth
trusting.

The other deliberate departure from "pure realism": the injected fraud
rate is `0.08` (8% of users get an episode), far above real-world fraud
prevalence. Run at a realistic sub-1% attack rate, a portfolio-scale
population (1,500 users, 60 days) produces so few positive examples that
the chronological test split ends up with well under a hundred fraud
transactions — not enough for the reported metrics to mean much of
anything. Real fraud teams solve this with far larger populations and
longer windows; here, the honest trade-off is to dial the rate up to get a
stable evaluation, and say so plainly rather than pretend the number would
hold at realistic prevalence.

## Why Flink isn't on the synchronous decision path

The architecture diagram draws one line from Kafka through Flink to the
decision. In this implementation that's true of the *system*, but not of
any single request's latency: `src/serving/api.py` and
`src/serving/stream_scorer.py` read and write Redis (`rt:*` keys)
synchronously, and Flink's job (`src/streaming/flink_job.py`) is a second,
independent consumer of the same Kafka topic that writes its own windowed
aggregates to a separate Redis namespace (`agg:*`).

That split is deliberate, not a shortcut. Flink's whole value is
*correctness over a window* — waiting out a configured watermark slack so
that a transaction arriving 20 seconds late still lands in the right
1-minute bucket. Waiting is exactly what a live risk decision cannot do; a
decision made 30 seconds after the transaction it was for is not a
real-time decision. So this system uses each tool for what it's actually
good at:

- **The synchronous path** (Redis, updated on every event as it's scored)
  answers "what does this user's history look like right now" with no
  window-closing delay, at the cost of only ever seeing one event's worth
  of update at a time.
- **Flink** answers "what did this user's traffic look like over this
  exact 60-second window, correctly accounting for anything that arrived
  out of order" — a question that's inherently about a batch of events, and
  inherently fine to wait a watermark's worth of time for. That output is
  suited to monitoring and to lower-urgency, longer-baseline features (the
  kind Feast's feature view demonstrates), not to unblocking a single
  transaction.

The Kafka producer (`src/producers/kafka_producer.py`) deliberately
delivers ~3% of events late (`--late-rate`) specifically so this
distinction shows up in practice, not just in a diagram: those events still
land in the correct Flink window (inside the watermark slack) or get routed
to the late-data side output (outside it) — either way, handled explicitly
rather than silently corrupting an already-closed window or being dropped
without a trace.

## Why two feature systems (Redis + Feast)

It would be simpler to have Feast own the whole online feature store and
retire `feature_store_client.py`. That doesn't fit the actual constraint:
Feast's online store is populated by `materialize`, a periodic batch
operation. The features this model actually depends on most
(`seconds_since_last_event`, `txn_count_5m`) change with every single
transaction — materializing on that cadence isn't "slower", it's a
different kind of feature entirely.

So they're kept as two honestly-different systems instead of forcing one
tool to pretend to be the other:

- `src/features/feature_store_client.py` — hand-rolled, synchronous,
  updated on every event, backed by Redis DB 0. This is what the model
  actually reads at scoring time.
- `feature_repo/` — one Feast feature view, `user_daily_stats`, backed by a
  daily-grained batch table and Feast's own Redis online store (DB 1, same
  container, different logical namespace so the two can never collide).
  This exists specifically to demonstrate the pattern real systems use for
  slower-moving baselines: `scripts/demo_feast_point_in_time.py` shows the
  same user's `txn_count_30d` answered two different, both-correct ways —
  "as of their transaction on day X" (an offline point-in-time join) versus
  "as of the most recent materialization" (the online snapshot) — which is
  the entire point of a feature store's point-in-time correctness
  guarantee, made concrete with real numbers instead of asserted in the
  abstract.

Feast is intentionally *not* wired into the training pipeline's main
feature matrix. `temporal_features.py` is deliberately the single source of
truth for what the model actually sees (see the leakage section above) —
folding a second, independently-computed feature source into that matrix
would make it much harder to keep the "every feature is causal and
tested" guarantee airtight, for a feature view that exists to demonstrate a
pattern rather than to move the model's metrics.

## The chronological split, and why it matters more than it sounds

`chronological_split()` cuts train/validation/test by wall-clock time, not
by shuffling rows into buckets. Concretely: if the dataset spans 60 days,
the first 60% of that *time range* is train, the next 20% is validation,
and the last 20% is test — regardless of how many events happen to fall in
each window (transaction volume isn't constant, so this isn't the same as
"the first 60% of rows").

Why this matters more than a one-line description suggests: a randomly
shuffled split lets information leak backward in time in a way that's easy
to miss. If user X has a fraud episode on day 10 and an ordinary
transaction on day 40, a random split might put the day-40 transaction in
training and the day-10 one in test — meaning the model was trained on data
from *after* the moment it's being asked to predict. Every rolling feature
computed for that test-set transaction would, in a real deployment, only
ever see the past — but the model that scored it would have been fit on
data from that user's future. The reported metrics would be real numbers
computed correctly on paper, describing an evaluation no live system could
ever reproduce.

The operating threshold has the same discipline applied one level up:
`pick_threshold()` picks the score cutoff that maximizes F1 on the
*validation* set, and that fixed threshold — not a new one refit for the
occasion — is what gets applied to the test set for the final reported
numbers. Tuning the threshold on the same data it's being scored against
would be the same leakage mistake, just moved from feature engineering to
decision-making.

## The decision engine's reason codes

Reason codes are evaluated independently of the model, from simple
feature-threshold rules (`configs/decision_policy.yaml`), rather than by
inspecting the model's internals (SHAP values, feature attributions).
That's a scope decision, not an oversight: SHAP would give more precise,
per-feature attribution for *why the model scored this transaction the way
it did*, but the rules give something a model-attribution approach
can't — a fixed, auditable vocabulary ("NEW_DEVICE", "HIGH_VELOCITY") that
means the same thing on every decision regardless of which model version
produced the score, and that a compliance reviewer can read without
knowing anything about XGBoost. Real fraud systems tend to want both;
this one implements the cheaper, more portfolio-appropriately-scoped half,
and the model's own risk score is still reported alongside it in every
decision record for whoever wants the fuller picture.

One consequence worth calling out because it's a real design choice, not
an accident: reason codes describe what was *observed*, independent of
what was *decided*. A transaction can be `ALLOW`ed and still carry a
`NEW_DEVICE` code, if the model's overall risk score stayed low despite
that one signal firing — informational, for whoever reviews the log later,
rather than only ever explaining REVIEW/BLOCK outcomes. The
`fallback_reason_code` (`MODEL_HIGH_RISK`) is the one piece that *is*
decision-gated: it only appears when a REVIEW/BLOCK fired with no matching
rule at all, meaning the model caught a combination of signals none of the
individual thresholds describe on their own — worth flagging as
unexplained-by-rules specifically, rather than mixing it in with the
informational codes.

## What broke during development, and what that says about the design

Kept here deliberately rather than edited away, because what breaks (and
how) says as much about a system's design as what works on the first try:

- **`np.timedelta64` vs. `pd.Timedelta` arithmetic.** Early
  `temporal_features.py` mixed the two; subtracting a `pd.Timedelta` from a
  `numpy.datetime64` scalar silently produces a `pandas.Timestamp` rather
  than another `datetime64`, which then fails inside `np.searchsorted` with
  an opaque `'<' not supported between instances of 'int' and 'Timestamp'`.
  Fixed by keeping `WINDOWS` as `np.timedelta64` throughout, so every
  operation on the time arrays stays in one type system.
- **A z-score with 1-2 samples.** `amount_zscore_30d` from a single prior
  transaction has `std = 0`, so the score divides by (near) zero and
  produces enormous, meaningless values (an early run showed a mean
  z-score of ~590,000 for ordinary transactions). Fixed with a minimum
  sample count (`>= 3`) and a floor on the standard deviation
  proportional to the mean, so the feature reads as "no reliable baseline
  yet" instead of a numeric artifact.
- **`docker-compose.yml` mounting a bind mount under another read-only
  one.** Grafana's compose service tried to mount the dashboard JSON
  directory as a second volume nested inside the already-mounted, read-only
  `provisioning/` tree — Docker can't create a mountpoint under a
  filesystem it's already mounted read-only, so the container failed
  immediately with `error mounting ... read-only file system`. Fixed by
  moving the dashboard JSON to live inside the same host directory that's
  already mounted, instead of adding a second overlapping mount.
- **`NaN` vs. `null` across the producer/consumer boundary.** The Kafka
  producer serializes each row of the historical parquet file with
  `pandas.Series.to_dict()`; a login/auth-failure row's missing `amount` is
  represented by pandas as float `NaN`, not `None`. `json.dumps` happily
  emits the non-standard `NaN` token for it, and — more importantly —
  Pydantic's `Field(ge=0)` constraint on `TransactionEvent.amount` rejects
  it outright, since `NaN` compares false to everything, including `>= 0`.
  The result, caught only by actually running the full docker-compose
  stack rather than by any unit test: **every single login and
  authentication-failure event was silently dropped as "malformed"** by
  the streaming consumer, which would have quietly starved several features
  (auth-failure counts, anything relying on non-transaction event history)
  of real data while looking, from the logs alone, like a minor warning
  rather than a systemic gap. Fixed in `_serialize()` with
  `row.where(pd.notna(row), None)` before building the JSON payload, and
  covered by a regression test
  (`test_serialize_converts_pandas_nan_to_json_null_not_the_nan_token`).
  This is the strongest argument in this whole project for why "the tests
  pass" and "the pipeline works end to end" are different claims — the unit
  tests around `_serialize` were passing the whole time; only running the
  real docker-compose stack and reading the consumer's own logs surfaced
  the gap.

## What was actually verified live, and what wasn't

Everything in this system was run for real at least once, with one
exception, called out plainly rather than left ambiguous:

**Verified running, end to end, via `docker compose up`:** Kafka (KRaft
mode), Redis, the FastAPI service, the Kafka producer, and the Kafka
streaming consumer. Concretely — not "the containers started", but the
actual data flow: the producer connected to Kafka and streamed real events;
the streaming consumer read them, scored real transactions, and wrote real
`ALLOW`/`REVIEW`/`BLOCK` decisions to `data/decision_log.jsonl` (10,000+
lines from one test run); the FastAPI `/score` endpoint was hit with real
HTTP requests and returned real decisions against the same Redis a
container was writing to. Prometheus was confirmed scraping both services'
`/metrics` (`"health":"up"` on both targets), and Grafana was confirmed
serving the provisioned dashboard against live data — the exact PromQL
queries in `monitoring/grafana/dashboards/decision_engine.json` were run
directly against Prometheus's API and returned real, non-zero rates once
the producer was streaming. Feast's point-in-time demo
(`scripts/demo_feast_point_in_time.py`) was run against a real Redis
online store, materialized from the real dataset, and produced the
concrete before/after numbers quoted [above](#why-two-feature-systems-redis--feast).

This is also how the NaN-serialization bug described above was actually
caught — not by a unit test, but by reading the streaming consumer's own
logs while it processed a real backlog of live-produced messages.

**Not verified live: the Flink job** (`src/streaming/flink_job.py`).
Building the custom Flink image (Java 11 + apt-installed Python 3 +
`apache-flink` pip package + the Kafka connector jar) inside the sandbox
this was developed in repeatedly exceeded what its build resources could
finish in reasonable time — not a code problem, a "cross-building a JVM
image with a pinned Python ML-adjacent package inside a constrained sandbox
takes longer than the sandbox allows" problem. What *is* true about that
file: it imports cleanly, its syntax was checked directly
(`ast.parse`), and every non-trivial API it uses
(`KafkaSource.builder()`, `WatermarkStrategy.for_bounded_out_of_orderness`,
`TumblingEventTimeWindows`, `.allowed_lateness()` +
`.side_output_late_data()`, a `RichMapFunction`'s `open()`/`close()` for a
long-lived Redis connection) is a stable, long-standing part of the
PyFlink DataStream API used the way its own documentation uses it — not
guessed at. If you build and run it yourself
(`docker compose up --build flink-jobmanager flink-taskmanager`, then
submit the job as the README describes) and something doesn't match this
description, that's the one component in this repo I'd genuinely want to
know about rather than confidently vouch for sight-unseen.

## What I'd do differently at real scale

- **SHAP-based reason codes** alongside the rule-based ones, so an
  explanation can point at "these five features together" instead of only
  "this one threshold crossed."
- **A real feature-freshness / data-quality monitor** — this system logs
  feature-store errors but doesn't yet alert on, say, a device-set that's
  suspiciously never growing, or a Redis TTL expiring mid-window.
  Monitoring in this project focuses on the decision engine's own output
  (which is what the brief asked for); production-grade input monitoring
  would be a project on its own.
- **Kafka/Redis persistence and clustering.** Both run as uncomplicated
  single nodes here — entirely appropriate for the scale and purpose of a
  portfolio project, entirely inappropriate for a system holding real
  financial decisions.
- **A real retraining pipeline** with drift detection deciding *when* to
  retrain, rather than the current manual, reproducible-but-manual
  `scripts/train_model.py`.
