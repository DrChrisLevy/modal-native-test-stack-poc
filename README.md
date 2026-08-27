> **This is a vibe-coded proof of concept built to show off some cool Modal features.**
>
> It is an experiment and demonstration—not a production framework or reference
> architecture.

# Modal-Native Test Stack POC

A clean-room, production-shaped multimodal application whose development and test
environment runs on Modal.

The laptop is only the control plane: edit files, run the lightweight CLI, and inspect
artifacts. FastAPI, Python/ML dependencies, real Hugging Face inference, PostgreSQL,
Redis, OpenSearch, pytest, Ruff, remote shells, and optional coding agents all execute
in Modal Sandboxes.

There is no Dockerfile, Docker daemon, Docker Compose, Docker-in-Docker, or GitHub
Actions workflow in this repository.

## What this demonstrates

- One `uv.lock`-backed Modal Image containing a substantial CPU ML stack.
- Source added after dependencies, so an application edit does not reinstall Torch.
- Seven real, immutable Hugging Face model revisions across text, vision, and audio.
- Model weights stored separately in a named Modal Volume and mounted read-only during
  application and test sessions.
- PostgreSQL, Redis, and OpenSearch as native Sandbox Sidecars on a private bridge
  network.
- Fresh, isolated service state for every test shard.
- Learned-duration, whole-file test fan-out; concurrent linting; merged coverage;
  JUnit output; phase timings; and guaranteed teardown.
- The same environment reused for tests, an API server, a remote shell, or a coding
  agent.
- Model and service paths tested for real. No fake model implementation or mocked model
  output is substituted into the application.

```text
local editor + Modal CLI
          |
          v
cached dependency Image + current source
          |
          +---- fresh test Sandbox 1 ---- PostgreSQL Sidecar
          |             |                Redis Sidecar
          |             |                OpenSearch Sidecar
          |             +---- /models (read-only Volume)
          |
          +---- fresh test Sandbox N ---- its own three Sidecars
          |
          +---- lint Sandbox (no services, weights, secrets, or egress)
```

Sidecars cannot mount Modal Volumes, so inference stays in the main Sandbox and only
the disposable infrastructure processes run as Sidecars.

## Real models

`models.lock.json` pins every repository to a full commit SHA.

| Capability | Hugging Face model | Output |
|---|---|---|
| Text embedding | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | normalized 384-d vector |
| Sentiment | [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | label and probability |
| Named entities | [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) | grouped spans, labels, and scores |
| Summarization | [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) | deterministic short text |
| Image classification | [`microsoft/resnet-18`](https://huggingface.co/microsoft/resnet-18) | ranked ImageNet labels |
| Image embedding | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | normalized 512-d vector |
| Speech recognition | [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en) | English transcript |

The seeder downloads only the artifacts required by those exact revisions. Six models
use safetensors; the pinned CLIP snapshot has no safetensors file and retains its
required `pytorch_model.bin`. The filtered Volume payload is about 1.9 GB instead of
about 8.4 GB of duplicate framework/export artifacts.

Normal sessions set both `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Loaders receive
only local `/models/<capability>` paths with `local_files_only=True` and
`trust_remote_code=False`. A missing snapshot fails visibly rather than downloading
during a test.

## Prerequisites

- `uv` on the local machine.
- A configured Modal profile and Environment.
- A Modal workspace allowlisted for Sandbox Sidecars, which are currently alpha.

Select the account outside the repository. For example:

```bash
export MODAL_PROFILE=your-profile
export MODAL_ENVIRONMENT=dev
```

No account name, token, secret, or environment-specific URL is committed here.

## First run

From the repository root:

```bash
# Prebuild the Python/ML runtime and all three Sidecar Images.
uv run modal-native-test-stack-poc build

# Populate the named model Volume from immutable revisions.
uv run modal-native-test-stack-poc seed

# Inspect the Volume's committed manifest without loading a model.
uv run modal-native-test-stack-poc check-models

# Fan the entire suite across four isolated full-stack Sandboxes.
uv run modal-native-test-stack-poc test --shards 4
```

The first complete passing suite calibrates file durations. Repeat the command to use
the learned plan. Preserve `artifacts/modal/test-durations.json` in any ephemeral CI
cache if duration learning should survive between runners.

`uv run` installs only the small local control-plane dependencies. The `remote` and
`test` extras are installed inside the cached Modal Image; application code and tests
are not executed on the laptop.

## Commands

### Images and weights

```bash
uv run modal-native-test-stack-poc build
uv run modal-native-test-stack-poc build --agent
uv run modal-native-test-stack-poc seed
uv run modal-native-test-stack-poc check-models
```

Use `build --force` only when intentionally bypassing Modal's Image cache. Use
`seed --force` only when intentionally replacing snapshots within this project's
named Volume.

### Tests

```bash
# Full suite, lint, JUnit, and merged coverage.
uv run modal-native-test-stack-poc test --shards 4

# Count-balanced control run without changing the learned history.
uv run modal-native-test-stack-poc test --shards 4 --scheduler count --no-learn

# Real-model, live-service, and end-to-end contracts only, still fanned out.
uv run modal-native-test-stack-poc smoke --shards 3

# Focused diagnosis in one fresh full-stack Sandbox.
uv run modal-native-test-stack-poc test --no-lint -- tests/model/test_text_models.py -x
```

The runner first collects node IDs remotely and preserves each test file as one
scheduling unit so module/session model reuse remains useful. The default `duration`
scheduler uses deterministic longest-processing-time planning against observed
whole-file runtimes. With no history, its fallback is equivalent to balancing test
counts. A complete passing run writes a versioned duration cache to
`artifacts/modal/test-durations.json`; later runs update it with an exponentially
weighted moving average so one cold-model outlier does not immediately rewrite the
plan. Focused and smoke runs never update the full-suite history. Use `--no-learn` to
freeze the cache for reproducible comparisons, or `--scheduler count` as the control.

Every test Sandbox gets its own PostgreSQL, Redis, and OpenSearch processes. Sidecars
are attached sequentially because the API is alpha; semantic readiness checks then run
concurrently.

Readiness means more than an open port:

- PostgreSQL executes a real query.
- Redis completes a write/read/delete round trip.
- OpenSearch creates, writes, reads, and deletes a temporary index.

Each shard emits JUnit and parallel coverage data. The coordinator combines coverage in
one remote Sandbox, enforces the configured application threshold, and copies final
artifacts to `artifacts/modal/<run-id>/`.

Every full run also records `summary.json`, `shard-plan.json`, the history used to plan
the run, and the newly observed file durations. The summary separates Image resolution,
Sandbox creation, Sidecar attachment, semantic readiness, collection/planning,
tests/lint, coverage/artifacts, and teardown submission. This makes “startup” an
inspectable sequence rather than a single ambiguous number. See [BENCHMARKS.md](BENCHMARKS.md)
for the frozen-history comparison used to validate the scheduler.

### Remote development shell

```bash
uv run modal-native-test-stack-poc shell
uv run modal-native-test-stack-poc shell --command 'pytest tests/model/test_text_models.py -x'
uv run modal-native-test-stack-poc shell --no-services --command 'ruff check src tests'
```

The default shell has real services and the read-only model Volume. Public network
egress is blocked unless `--network` is explicitly supplied.

### FastAPI

```bash
uv run modal-native-test-stack-poc api
```

This starts Uvicorn in the prepared Sandbox, prints the encrypted Modal tunnel URL, and
attaches a remote shell. Exiting the shell tears down the API and all three Sidecars.

Principal routes include:

```text
GET  /health/live
GET  /health/ready
GET  /v1/models
POST /v1/text/embed
POST /v1/text/sentiment
POST /v1/text/entities
POST /v1/text/summarize
POST /v1/images/classify
POST /v1/images/embed
POST /v1/audio/transcribe
POST /v1/assets/text
POST /v1/assets/image
POST /v1/assets/audio
GET  /v1/assets/{asset_id}
POST /v1/search
```

Asset endpoints exercise the full path: inference, revision-aware Redis caching,
PostgreSQL persistence, and OpenSearch lexical/vector indexing.

### Coding-agent environment

The optional agent Image installs a checksum-verified, version-pinned Codex CLI after
the dependency layer and before changing source. It uses the same models and Sidecars
as tests.

For a one-shot Codex task, expose an API key through a named Modal Secret rather than a
file in the checkout:

```bash
export MODAL_NATIVE_TEST_STACK_POC_AGENT_SECRET=your-modal-secret-name
uv run modal-native-test-stack-poc agent --prompt 'Inspect the failing tests, fix the bug, and rerun them.'
```

An interactive session omits `--prompt`. A custom command can demonstrate the reusable
agent environment without building Codex:

```bash
uv run modal-native-test-stack-poc agent --command 'python -c "print(\"agent environment ready\")"'
```

The checkout is initialized as an ephemeral Git repository and the resulting binary
patch, including new files, is copied to `artifacts/modal/<run-id>/agent.patch`. The
local checkout is never changed automatically.

## Resource ownership and cleanup

The harness uses deliberately narrow names and tags:

- App: `modal-native-test-stack-poc`
- Volume: `modal-native-test-stack-poc-models`
- Sandbox ownership tag: `modal-native-test-stack-poc-owner=modal-native-test-stack-poc`
- Per-run and per-role tags on every Sandbox

Normal commands terminate their main Sandboxes and Sidecars in `finally` blocks. If a
client is interrupted or a failed run was deliberately kept, inspect and clean only
this project's tagged resources:

```bash
uv run modal-native-test-stack-poc status
uv run modal-native-test-stack-poc cleanup
uv run modal-native-test-stack-poc cleanup --run-id <run-id>
```

The model Volume is intentionally persistent so subsequent runs do not redownload
weights. Its deletion is separate, explicit, scoped, and irreversible:

```bash
uv run modal-native-test-stack-poc delete-models --yes
```

Cached Images and the empty named App may remain as Modal metadata, but no live compute
is required between commands.

All names and resource sizes can be overridden with `MODAL_NATIVE_TEST_STACK_POC_*` environment
variables; see `src/modal_native_test_stack_poc/remote/config.py`.

## Alpha caveats

Modal Sandbox Sidecars are experimental and require workspace access. In the current
API:

- Sidecar Images must be built before attachment.
- Sidecars share the main Sandbox CPU and memory allocation.
- Sidecars cannot mount Volumes.
- Sidecar filesystems are ephemeral and are not included in Sandbox snapshots.
- Creation returns before a service is ready, so bounded readiness polling is required.

This project intentionally treats those constraints as visible application behavior,
not hidden implementation details.

## Clean-room boundary

The repository uses only public packages, public model repositories, generated media
fixtures, synthetic application code, and a freshly resolved lockfile. It does not
contain or mechanically reproduce private application source, tests, schemas, service
names, repository metadata, credentials, or dependency locks.

## License

The project source is MIT licensed. Model weights remain governed by their respective
Hugging Face repository licenses.
