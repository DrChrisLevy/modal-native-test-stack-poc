> **This is a vibe-coded proof of concept for running a Python application and its
> development workflow on Modal. It is not production software.**

# Modal-Native Test Stack POC

This repository demonstrates a Python development workflow that runs entirely on
Modal:

- A standalone FastAPI application exercises real inference and data services.
- A Modal runner builds the environment and runs tests, the API, shells, and coding
  agents in a Sandbox.

The application does not import Modal. The runner packages it into an Image, mounts
reusable data from a Volume, and attaches PostgreSQL, Redis, and OpenSearch Sidecars.

```text
local CLI -> Modal Sandbox -> application + Volume + Sidecars
```

There is no Docker, Compose, Docker-in-Docker, or GitHub Actions workflow.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A configured Modal profile and Environment
- Access to Modal Sandbox Sidecars, which are currently alpha

Select the Modal account before running commands:

```bash
export MODAL_PROFILE=your-profile
export MODAL_ENVIRONMENT=dev
```

## Setup

```bash
uv run modal-native-test-stack-poc build
uv run modal-native-test-stack-poc seed
uv run modal-native-test-stack-poc test
```

`build` prepares the Images, `seed` populates the reusable Volume, and `test` creates
one Sandbox with one set of Sidecars. pytest and Ruff run remotely, and test artifacts
are copied to `artifacts/modal/<run-id>/`.

Tests do not run locally. Pass pytest arguments after `--`:

```bash
uv run modal-native-test-stack-poc test --workers 1 --no-lint -- \
  tests/model/test_text_models.py -x
```

## Other commands

```bash
uv run modal-native-test-stack-poc smoke
uv run modal-native-test-stack-poc shell
uv run modal-native-test-stack-poc api
uv run modal-native-test-stack-poc agent --secret your-openai-secret
uv run modal-native-test-stack-poc status
uv run modal-native-test-stack-poc cleanup
```

Use `<command> --help` for options. `api` prints a tunnel URL; `shell` and `agent` open
interactive remote sessions.

## Cleanup

Sandboxes and Sidecars terminate when a command exits. Images and the Volume persist.

To delete the Volume:

```bash
uv run modal-native-test-stack-poc delete-models --yes
```

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for measured serial and parallel test timings.

The project source is MIT licensed. Dependencies and model data retain their own
licenses.
