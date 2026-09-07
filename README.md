# Real-Time Transaction Risk & Decision Engine

A distributed, real-time ML system that scores every incoming transaction
for fraud risk and turns that score into an **ALLOW / REVIEW / BLOCK**
decision, end to end: synthetic transactions → Kafka → Flink → Redis →
model inference → decision engine → monitoring.

It is a portfolio project, not a production banking platform — every piece
is real and connected, scaled to run comfortably on a laptop.

```
Synthetic Transactions
        |
      Kafka
        |
   Flink Streaming (event-time windows, watermarks, late data)
        |
 Real-Time Features (Redis)  <---- also read/written synchronously by --,
        |                                                                |
   ML Inference (XGBoost)                                                |
        |                                                                |
  Decision Engine (config-driven thresholds + reason codes)              |
   /      |       \                                                     |
ALLOW   REVIEW   BLOCK                                                   |
        |                                                                |
   Prometheus + Grafana                                                  |
                                                                          |
   FastAPI  /score  ------------------------------------------------------
```

## The problem

A payment processor has milliseconds to decide whether to let a transaction
through. Getting it wrong in either direction is expensive: block too much
and you lose legitimate customers; block too little and you eat fraud
losses. That decision can't be made from the transaction alone — "$4,500 at
2am" is unremarkable for one user and wildly out of character for another.
It has to be made from **that user's own recent behaviour**: how often they
transact, what devices they use, whether they just failed a few logins.

That behavioural context is what makes this a *real-time* ML problem rather
than an ordinary batch classification one. The features that actually
matter — "3 transactions in the last 5 minutes", "a device this user has
never used" — describe a rolling window of state that changes with every
event. A batch job that recomputes them once a day would be scoring against
information that's already stale by the time a live transaction needs a
decision. That's the whole reason this system exists as a streaming
pipeline with an online feature store, instead of a notebook that outputs a
CSV of predictions.

## Architecture and data flow

| Stage | What it does | Where |
|---|---|---|
| **Data generation** | Builds a population of synthetic users, each with a persistent identity — spending habits, usual devices, active hours — and simulates their event stream (transactions, logins, auth failures) over 60 days, with realistic fraud episodes spliced in. | `src/data/`, `src/producers/` |
| **Kafka** | Carries the transaction stream. The producer replays the historical dataset in accelerated wall-clock time, deliberately delivering a few percent of events late. | `src/producers/kafka_producer.py` |
| **Flink** | Consumes the stream with event-time watermarks and 1-minute tumbling windows per user, handling out-of-order and late-arriving events explicitly rather than assuming order. Writes windowed aggregates to Redis. | `src/streaming/flink_job.py` |
| **Redis** | Two roles: (1) the **online feature store** the decision path reads/writes synchronously on every transaction (`rt:*` keys), and (2) the target Flink's windowed aggregates land in (`agg:*` keys) — see [Why Flink isn't on the decision path](docs/understanding.md#why-flink-isnt-on-the-synchronous-decision-path). | `src/features/feature_store_client.py` |
| **Feast** | A small, separate feature view demonstrating point-in-time-correct offline retrieval against an online snapshot — a slower-moving, batch-computed baseline layered on top of the fast Redis path, not a replacement for it. | `feature_repo/` |
| **ML inference** | An XGBoost classifier trained on causally-computed features, loaded once and reused for every score. | `src/models/` |
| **Decision engine** | Config-driven thresholds turn a risk score into ALLOW/REVIEW/BLOCK, plus human-readable reason codes from independent feature-threshold rules. | `src/decisions/` |
| **Serving** | Two equivalent entry points into the same scoring/decision code: a synchronous FastAPI endpoint, and a Kafka-driven streaming consumer. | `src/serving/` |
| **Monitoring** | Prometheus scrapes both serving paths; Grafana visualises decision rates, latency, and the risk-score distribution. | `monitoring/` |

## The model

**Features are temporal and strictly causal.** Every input
(`src/features/temporal_features.py`) is computed from a user's own event
history *strictly before* the transaction being scored — rolling
transaction counts and amount statistics over 5-minute/1-hour/24-hour/30-day
windows, time since the last event, whether the device has been seen
before, recent authentication failures, distinct locations seen recently.
None of it depends on hidden profile labels, suspicion scores, or anything
that happens after the transaction. See
[docs/understanding.md](docs/understanding.md#avoiding-leakage) for how this
is tested, not just claimed.

**The label comes from injected fraud episodes**
(`src/producers/fraud_injection.py`), not from the features themselves — two
realistic patterns (account takeover, card testing) spliced into otherwise
normal user timelines, the same way a real system trains on confirmed-fraud
labels while scoring on independently engineered signals.

**The split is chronological.** Train/validation/test are cut by time, not
by shuffling rows — a model is only ever evaluated on transactions that
happened after everything it was trained on, which is the only way the
reported metrics mean what a live deployment would actually see.

Current test-set results (`models/risk_model_metadata.json` after training):

| Metric | Value |
|---|---|
| ROC-AUC | ~0.9998 |
| PR-AUC | ~0.96 |
| Precision | ~0.93 |
| Recall | ~0.89 |
| F1 | ~0.91 |

(Exact numbers regenerate with `scripts/train_model.py` and will vary
slightly run to run only if the data or code changes — the random seeds are
fixed.)

## The decision engine

`configs/decision_policy.yaml` is data, not code:

```
risk <  thresholds.allow_below       -> ALLOW
thresholds.allow_below <= risk <
  thresholds.block_at_or_above       -> REVIEW
risk >= thresholds.block_at_or_above -> BLOCK
```

Reason codes (`NEW_DEVICE`, `HIGH_VELOCITY`, `AMOUNT_ANOMALY`,
`RECENT_AUTH_FAILURES`, `MULTIPLE_LOCATIONS`) are evaluated independently of
the model score, from simple feature-threshold rules, so a decision is
explainable without inspecting the model. Every decision — event ID, user
ID, risk score, decision, reason codes, timestamp — is appended to
`data/decision_log.jsonl`.

## Monitoring

Both serving paths (`src/serving/api.py`, `src/serving/stream_scorer.py`)
expose Prometheus metrics under the same names (`src/serving/metrics.py`):
decisions by outcome, inference latency, the risk-score distribution, raw
events processed, and feature-store errors. The Grafana dashboard
(`monitoring/grafana/dashboards/decision_engine.json`) is provisioned
automatically — open it and it's already there.

## Running it locally

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install pytest httpx fakeredis  # for the test suite
```

### 2. Generate data, train, test

```bash
python -m scripts.generate_historical_data   # data/historical_events.parquet
python -m scripts.train_model                # models/risk_model.joblib + metadata
pytest -q                                    # 60+ tests; trains a small model itself if none exists yet
```

### 3. Try the decision engine without any infrastructure at all

```bash
uvicorn src.serving.api:app --reload
```

```bash
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{
  "event_id": "evt_1", "user_id": "user_00001",
  "event_time": "2026-03-05T12:00:00Z",
  "amount": 15000, "device_id": "new_device_1", "location": "Mumbai"
}'
```

This needs Redis reachable at `REDIS_URL` (defaults to
`redis://localhost:6379/0`) — the quickest way is
`docker run -p 6379:6379 redis:7-alpine`.

### 4. The full stack

```bash
docker compose up --build
```

Brings up Kafka, Redis, a 2-node Flink cluster, the API, the Kafka producer
(replaying the dataset live), the streaming consumer, Prometheus, and
Grafana.

| Service | URL |
|---|---|
| API | http://localhost:8000/docs |
| Flink dashboard | http://localhost:8081 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (anonymous viewer access) |

The Flink job itself isn't auto-submitted (its exact behaviour is worth
watching land, not hidden behind compose startup):

```bash
docker compose exec flink-jobmanager \
  flink run -py /opt/flink-job/flink_job.py
```

### 5. Feast — point-in-time correctness

Run from the host venv (Feast here is a one-off demonstration, not a
standing service — see [docs/understanding.md](docs/understanding.md)):

```bash
python -m scripts.build_feast_offline_source
cd feature_repo && ../.venv/bin/feast apply && cd ..   # (Scripts\feast.exe on Windows)
python -m scripts.materialize_feast
python -m scripts.demo_feast_point_in_time
```

## Project layout

```
src/
  data/            synthetic user population + historical event simulation
  producers/       event schema, event generation, fraud injection, Kafka producer
  features/        causal temporal features + the Redis online feature store client
  models/          training, evaluation, and the inference interface
  decisions/       the decision engine, its policy config, and decision logging
  streaming/       the PyFlink job
  serving/         FastAPI app, Kafka streaming consumer, Prometheus metrics
feature_repo/      Feast entities, feature views, feature_store.yaml
configs/           decision_policy.yaml
monitoring/        Prometheus config, Grafana provisioning + dashboard
docker/            Dockerfiles for the app image and the Flink image
scripts/           CLI entry points for every step above
tests/             60+ tests, including online/offline feature parity and no-leakage checks
docs/understanding.md   design decisions, trade-offs, and the interview-style walkthrough
```

## What's deliberately out of scope

This is a portfolio-scale system, not a production one, on purpose:

- No auth/multi-tenancy on the API.
- No model retraining pipeline — training is a manual, reproducible script.
- No Kafka/Redis persistence tuning or clustering — single-node everything.
- Feast runs one feature view, not a catalog — it exists to demonstrate the
  offline/online/point-in-time pattern, not to be a complete feature
  platform.

See [docs/understanding.md](docs/understanding.md) for the reasoning behind
these and the bigger design decisions.
