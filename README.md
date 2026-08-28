> **This is a vibe-coded proof of concept for running a Python application and its
> development workflow on Modal. It is not production software.**

# Intro

This repo goes along with my [blog post](https://drchrislevy.com/blog/) about Modal's
[Sandbox Sidecars](https://modal.com/docs/guide/sandbox-sidecars), which were in alpha at
the time of writing.

It runs a dummy FastAPI/ML application, PostgreSQL, Redis, OpenSearch, pytest, remote
shells, and Codex on Modal—without Docker Compose, Docker-in-Docker, or GitHub Actions.

The code could possibly be trash. I never read it. It's just a talking point for my blog post
and to give you ideas. The point is the potential of using these Modal primitives
for development, CI, experiments with agents, etc.

## Setup


```bash
export MODAL_PROFILE=your-profile
export MODAL_ENVIRONMENT=your-environment
```

```bash
uv sync --frozen
uv run modal setup
# Create once for the agent command (expects OPENAI_API_KEY in your local environment).
uv run modal secret create openai-secret OPENAI_API_KEY="$OPENAI_API_KEY"
```

```bash
# pre-builds the environment
uv run modal-native-test-stack-poc build
# download models to the Modal Volume
uv run modal-native-test-stack-poc seed
# Run the full pytest suite and Ruff remotely in one Modal Sandbox with PostgreSQL, Redis, and OpenSearch Sidecars.
uv run modal-native-test-stack-poc test
```


### Other commands

```bash
# open a shell in a fresh Modal Sandbox
uv run modal-native-test-stack-poc shell
# Start FastAPI with real models and Sidecars on Modal, print its HTTPS URL, and keep it running until you exit
uv run modal-native-test-stack-poc api
# launch an interactive Codex agent in the environment
uv run modal-native-test-stack-poc agent
# run one Codex prompt and exit
uv run modal-native-test-stack-poc agent --prompt "Run the tests and summarize any failures."
# list running sandboxes
uv run modal-native-test-stack-poc status
# stop/kill running sandboxes
uv run modal-native-test-stack-poc cleanup
```

## Cleanup

Sandboxes and Sidecars terminate when a command exits. Images and the Volume persist.

To delete the Volume:

```bash
uv run modal-native-test-stack-poc delete-models --yes
```
