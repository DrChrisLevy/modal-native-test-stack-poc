> **This is a vibe-coded proof of concept for running a Python application and its
> development workflow on Modal. It is not production software.**

# Intro

This repo goes along with my [blog post](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fmodal_native_test_stack%2Fmodal_native_test_stack.md) about Modal's
[Sandbox Sidecars](https://modal.com/docs/guide/sandbox-sidecars), which were in alpha at
the time of writing.

It runs a dummy FastAPI/ML application, PostgreSQL, Redis, OpenSearch, pytest, remote
shells, and Codex on Modal—without Docker Compose, Docker-in-Docker, or GitHub Actions.

The point is to demonstrate what these Modal primitives can make possible for development,
CI, and agent workflows—not to present production-ready application code.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/), a Modal account, and an OpenAI API key if
you want to run the agent commands.

```bash
uv sync --frozen
uv run modal setup
```

Create this Secret once if you want to run the agent command. It expects `OPENAI_API_KEY` in
your local environment:

```bash
uv run modal secret create openai-secret OPENAI_API_KEY="$OPENAI_API_KEY"
```

```bash
# Pre-build the runtime and Sidecar Images.
uv run modal-native-test-stack-poc build
# Download models to the Modal Volume.
uv run modal-native-test-stack-poc seed
# Run pytest and Ruff remotely in one full-stack Modal Sandbox.
uv run modal-native-test-stack-poc test
```


### Other commands

```bash
# Open a shell in a fresh Modal Sandbox.
uv run modal-native-test-stack-poc shell
# Start FastAPI and keep it running until you exit the attached shell.
uv run modal-native-test-stack-poc api
# Launch an interactive Codex agent in the environment.
uv run modal-native-test-stack-poc agent
# Run one Codex prompt and exit.
uv run modal-native-test-stack-poc agent --prompt "Run the tests and summarize any failures."
# List running Sandboxes.
uv run modal-native-test-stack-poc status
# Terminate running Sandboxes owned by the POC.
uv run modal-native-test-stack-poc cleanup
```

## Cleanup

Sandboxes and Sidecars normally terminate when a command exits. The exception is
`test --keep-on-failure`, which deliberately preserves a failed environment for debugging.
Unchanged Image layers can be reused from Modal's cache, and the named model Volume persists.

To delete the Volume:

```bash
uv run modal-native-test-stack-poc delete-models --yes
```

If you created the Secret only for this POC, you can also remove the remaining Modal objects:

```bash
uv run modal secret delete openai-secret --allow-missing --yes
uv run modal app stop modal-native-test-stack-poc --yes
```
