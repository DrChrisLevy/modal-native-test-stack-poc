# Benchmarks

Measurements were recorded on 2026-08-26 with Modal SDK 1.5.4 and CPython
3.12.10. Each run used:

- One Modal Sandbox with 4 CPU and 16,384 MiB memory
- PostgreSQL, Redis, and OpenSearch Sidecars
- Cached Images and a populated data Volume
- Real model inference
- Concurrent Ruff
- 94% configured application coverage

Three serial and three parallel runs were interleaved. No Sandbox from a previous
trial was running when the next trial started.

~~~bash
uv run modal-native-test-stack-poc test --workers 1
uv run modal-native-test-stack-poc test --workers 4
~~~

## Results

| Metric | Serial median (range) | Four-worker median (range) |
|---|---:|---:|
| End-to-end | 59.393s (58.261–64.992) | 48.589s (43.240–51.000) |
| Pytest | 39.424s (38.966–43.109) | 31.157s (26.336–31.174) |
| Startup | 19.330s (18.401–21.381) | 17.069s (16.469–19.504) |

Four workers reduced median end-to-end time by 18.2% and median pytest time by
21.0%.

| Trial | Workers | End-to-end | Pytest | Startup |
|---|---:|---:|---:|---:|
| S1 | 1 | 64.992s | 43.109s | 21.381s |
| X1 | 4 | 51.000s | 31.174s | 19.504s |
| S2 | 1 | 59.393s | 39.424s | 19.330s |
| X2 | 4 | 43.240s | 26.336s | 16.469s |
| S3 | 1 | 58.261s | 38.966s | 18.401s |
| X3 | 4 | 48.589s | 31.157s | 17.069s |

Startup includes Image resolution, Sandbox creation, Sidecar attachment, service
readiness, and schema bootstrap. End-to-end time also includes pytest, Ruff, artifact
collection, and teardown submission.

## Limits

- These measurements cover repeat runs with cached Images and populated data.
- Model inference is CPU-bound and varies between fresh processes.
- Sidecars share the Sandbox CPU and memory allocation.
- Three repetitions are sufficient for this POC, not for a general Modal latency
  claim.
- Sandbox Sidecars are an alpha feature.
