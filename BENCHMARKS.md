# Benchmarks

This file records measurements from the clean-room application. It is a POC benchmark
report, not a claim about every Modal workspace, region, or workload.

## Decision: one stack, capability-aware xdist

The adopted topology uses one fresh Modal Sandbox, one PostgreSQL/Redis/OpenSearch
Sidecar set, and four pytest-xdist workers inside the Sandbox. The workers are grouped
by text, image, audio, and non-model core tests so process-local lazy model loading is
reused instead of scattered randomly.

The comparison used the same application source, cached Images, populated model
Volume, service topology, and 429-test suite on both sides. All seven pinned Hugging
Face models performed real inference; no model outputs were mocked. Ruff ran
concurrently in the same Sandbox. Every scored run passed at 94% configured
application coverage.

Measurements were taken on 2026-08-26 with Modal SDK 1.5.4 and CPython 3.12.10. The
Sandbox used the project's 4 CPU and 16,384 MiB reservation, shared with its Sidecars.
A status check between trials confirmed there was no live compute from the previous
run.

```bash
# Serial control: one process in the same one-stack topology.
uv run modal-native-test-stack-poc test --workers 1

# Adopted topology.
uv run modal-native-test-stack-poc test --workers 4
```

Three serial and three grouped runs were interleaved after source and Image caches were
stable.

| Metric | Serial median (range) | Grouped xdist median (range) | Result |
|---|---:|---:|---:|
| End-to-end wall time | 59.393s (58.261–64.992) | 48.589s (43.240–51.000) | 18.2% lower |
| Pytest wall time | 39.424s (38.966–43.109) | 31.157s (26.336–31.174) | 21.0% lower |
| Startup through schema bootstrap | 19.330s (18.401–21.381) | 17.069s (16.469–19.504) | similar |
| Aggregate JUnit testcase time | 32.429s (32.175–35.927) | 67.945s (56.463–68.082) | overlaps across workers |

Raw scored trials:

| Trial | Workers | End-to-end | Pytest | Startup |
|---|---:|---:|---:|---:|
| S1 | 1 | 64.992s | 43.109s | 21.381s |
| X1 | 4 | 51.000s | 31.174s | 19.504s |
| S2 | 1 | 59.393s | 39.424s | 19.330s |
| X2 | 4 | 43.240s | 26.336s | 16.469s |
| S3 | 1 | 58.261s | 38.966s | 18.401s |
| X3 | 4 | 48.589s | 31.157s | 17.069s |

Aggregate JUnit time is not a wall-clock or cost metric under xdist: testcase clocks
overlap and each worker records its own process-local fixture setup. The pytest process
wall time is the relevant test critical path.

## Why it replaced six full stacks

The previous prototype planned whole test files from learned duration history and
created six full-stack Sandboxes. That introduced a duration cache, EWMA updates,
custom planning, cross-Sandbox JUnit/coverage handling, and 18 Sidecars per run. Its
best historical strategy was count balancing:

| Topology | End-to-end median | Test barrier median | Test Sandboxes | Sidecars |
|---|---:|---:|---:|---:|
| Adopted one-stack grouped xdist | 48.589s | 31.157s | 1 | 3 |
| Historical six-stack count plan | 54.973s | 29.256s | 6 | 18 |
| Historical six-stack duration plan | 61.425s | 31.900s | 6 | 18 |

The six-stack count plan made pytest about 1.9 seconds faster, but the adopted design
was about 6.4 seconds faster end to end while provisioning one-sixth as many complete
test stacks. The duration plan was slower on both measures. Historical runs used 427
tests rather than 429, so this table is a topology reference rather than a
resource-equivalent A/B test.

The simpler design also matches the workload better:

- The Sidecars are shared infrastructure, so paying their startup cost once is enough.
- Capability groups express the dominant scheduling fact directly: which model bundle
  should remain warm in which worker.
- A session registry is reused by direct model, API, and end-to-end tests in each
  capability process.
- UUID database rows, worker-scoped Redis namespaces, and unique OpenSearch indexes
  prevent shared-state collisions.
- pytest-cov combines xdist coverage natively in the one Sandbox.
- No historical timing state is required for a correct or fast run.

After this experiment the unused duration scheduler and its 47 harness-only tests were
removed. That changes the final suite count but removes no application, inference,
API, service, or end-to-end contract.

Two final cleanup validations passed the resulting 382-test suite at 94.68% coverage.
The dependency-rebuild run took 28.7 seconds in pytest and 88.9 seconds end to end,
including 42.4 seconds rebuilding the smaller Image. The immediately following
cached-Image run took 45.0 seconds in pytest and 64.9 seconds end to end. They are
reported as validation rather than added to the scored table because the suite and
dependency Image had changed. The spread is also a useful reminder that fresh-process
CPU model inference is variable; the decision above uses repeated medians, not a single
best run.

## What the timing includes

End-to-end time begins before configuration, model-manifest, App, and Image resolution.
It ends after teardown has been submitted. Each `summary.json` separates:

1. Image resolution.
2. Main Sandbox creation.
3. Sidecar attachment.
4. Semantic service readiness.
5. PostgreSQL schema bootstrap.
6. Pytest and concurrent Ruff.
7. Artifact collection.
8. Teardown submission.

“Startup” in the tables is phases 1–5. Readiness is behavioral, not a container health
check: PostgreSQL executes a query, Redis performs a round trip, and OpenSearch creates,
writes, reads, and deletes a temporary index.

## Interpretation limits

- These are fresh processes and fresh service state, but cached Images and a populated
  model Volume—the intended repeat-CI path, not a first-ever build or model download.
- Model loading is lazy and process-local, so group composition matters.
- Sidecars share the main Sandbox CPU and memory allocation.
- Three repetitions per topology are enough for this POC decision, not a statistically
  strong platform latency study.
- Sandbox Sidecars are an alpha Modal feature and their behavior may change.
