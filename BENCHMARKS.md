# Benchmarks

This file records public, reproducible measurements from the clean-room application.
It is a benchmark report, not a claim about every Modal workspace, region, or workload.

## Fresh-Volume adaptive-sharding experiment

The question was narrow: after the scheduler has learned whole-file runtimes from real
test runs, does its next fixed plan reduce the test critical path compared with
balancing files by collected test count?

The experiment used the same cached Images, freshly named populated model Volume,
application source, 427-test suite, and six fresh full-stack Sandboxes for every scored
run. The model data was copied between Volumes entirely inside Modal before the
experiment; this measured the normal populated-Volume CI path, not a first model
download. Each test Sandbox received its own PostgreSQL, Redis, and OpenSearch
Sidecars. A separate service-free Sandbox ran Ruff concurrently. All inference used
the seven real, revision-pinned Hugging Face models; no model output was mocked.

Measurements were taken on 2026-08-26 with Modal SDK 1.5.4 and CPython 3.12.10. Each
main Sandbox used the project's default 4 CPU and 16,384 MiB memory reservation, shared
with its Sidecars.

### Learning, then freezing

The previous duration file was moved aside. One count-balanced calibration run built a
fresh 24-file history from zero. Three successful duration-scheduled runs then updated
that history naturally using the scheduler's exponentially weighted moving average.
Those four runs were training observations, not scored trials.

The adapted history was snapshotted after the third learning run. All scored runs used
`--no-learn`, and the working history's checksum still matched the frozen snapshot
afterward. This separates adaptation from measurement while keeping the input plan
identical across repetitions.

```bash
# Natural learning passes (run three times after calibration)
uv run modal-native-test-stack-poc test --shards 6 --scheduler duration

# Frozen-history scored trials
uv run modal-native-test-stack-poc test --shards 6 --scheduler duration --no-learn
uv run modal-native-test-stack-poc test --shards 6 --scheduler count --no-learn
```

Scored runs alternated A/B for three repetitions. A status check between every trial
confirmed that the previous run had no live Sandboxes before the next run began.

| Metric | Duration-aware median (range) | Count-balanced median (range) | Result |
|---|---:|---:|---:|
| End-to-end wall time | 61.425s (54.536–71.763) | 54.973s (54.080–55.529) | duration 11.7% higher |
| Tests + concurrent lint barrier | 31.900s (24.727–32.556) | 29.256s (27.602–29.838) | duration 9.0% higher |
| Slowest test shard | 31.898s (24.726–32.556) | 29.254s (27.599–29.837) | duration 9.0% higher |
| Parallel test utilization | 70.47% (69.55–79.86%) | 54.31% (49.93–55.85%) | duration +16.16 points |
| Ruff process | 0.170s (0.161–0.262) | 0.162s (0.159–0.176) | effectively unchanged |

Raw scored trials:

| Trial | Scheduler | End-to-end | Test/lint barrier |
|---|---|---:|---:|
| A1 | duration | 54.536s | 24.727s |
| B1 | count | 54.080s | 29.838s |
| A2 | duration | 61.425s | 31.900s |
| B2 | count | 55.529s | 29.256s |
| A3 | duration | 71.763s | 32.556s |
| B3 | count | 54.973s | 27.602s |

The learned plan did what its cost model asked: median utilization rose from 54.31% to
70.47%. Count planning placed 69–73 tests on each shard, whereas duration planning
intentionally placed 4, 24, 15, 7, 226, and 151 tests on its six shards.

It did **not** win this frozen-history sample overall. A1 reduced the critical path by
5.111s relative to B1. In A2 and A3, real-model API and end-to-end pipeline tests ran
well above their learned weights, and the fixed plan could not adapt because learning
was deliberately disabled for measurement. The duration plan also performed more
aggregate JUnit testcase work: a 111.950s median versus 73.706s for count scheduling,
largely because spreading model-heavy files across processes duplicates model loading.

That is the useful result: duration history is a planning signal, not an oracle. It can
improve balance, but model affinity, process-local lazy loading, runtime drift, and
outliers also matter. In normal use, successful full-suite runs continue updating the
history; only the benchmark froze it.

All scored runs passed 427 tests, Ruff, combined coverage, and the 80% coverage gate;
measured configured application coverage was 94%. The coverage configuration excludes
the local CLI and most Modal orchestration modules.

## What the timing includes

End-to-end wall time begins before configuration, model-manifest, App, and Image
resolution. It ends after teardown has been submitted. Each `summary.json` separates:

1. Image resolution.
2. Main Sandbox creation.
3. Sidecar attachment.
4. Semantic service readiness.
5. Remote test collection and shard planning.
6. Concurrent tests and lint.
7. Coverage merge and artifact collection.
8. Teardown submission.

Across the six scored runs, startup through semantic readiness had a 22.200s pooled
median (18.565–25.801s). Readiness alone had a 14.047s median (11.282–17.674s), about
63% of startup. Other pooled phase medians were 2.342s for Image resolution, 2.892s
for Sandbox creation, 2.408s for Sidecar attachment, 2.161s for collection/planning,
3.146s for coverage/artifacts, and 0.384s for teardown submission.

The test critical path is the longest pytest process wall time, not the sum of JUnit
testcase durations. Parallel utilization is the sum of shard process times divided by
the idealized capacity of `shard_count * slowest_shard_seconds`.

## Interpretation limits

- These were fresh processes and fresh service state, but cached Images and an already
  populated model Volume. This is the normal CI path, not a first-ever dependency build
  or model download.
- Model loading is lazy and process-local. A file can be much faster when a preceding
  file has already loaded the same model, so runtimes depend partly on placement. The
  scheduler keeps files intact and smooths observations, but its cost model does not
  explicitly represent model affinity.
- Every six-shard trial provisioned six complete service stacks—18 Sidecars total.
  Capability-aware provisioning is a separate optimization and was not mixed into this
  comparison.
- Service readiness and real inference varied more than lint, which is why scheduler
  behavior is clearest in the test critical path rather than end-to-end wall time.
- Three repetitions per strategy are enough for a POC demonstration, not a general
  Modal latency study or a statistically strong performance claim.

## One-shard reference

A separate one-shard run passed the same 427 tests at 94% configured application
coverage. It took 71.206s end to end, including 23.182s through semantic readiness and
44.037s for tests plus concurrent lint.

Relative to that single observation, the six-shard count median reduced the test
barrier by 33.6% and end-to-end time by 22.8%. The six-shard duration median reduced
the barrier by 27.6% and end-to-end time by 13.7%.

The one-shard run is a topology reference, not part of the alternating A/B comparison.
It shows the tradeoff between model reuse within one process and parallel fan-out; a
different shard topology can change model-loading work as well as concurrency.
