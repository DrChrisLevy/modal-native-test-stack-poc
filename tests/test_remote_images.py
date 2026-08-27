from __future__ import annotations

from modal_native_test_stack_poc.remote.images import (
    CODEX_RELEASE_TAG,
    REMOTE_VIRTUAL_ENV,
    _codex_install_command,
    _login_shell_profile_command,
)


def test_login_shells_keep_the_remote_virtual_environment_on_path() -> None:
    command = _login_shell_profile_command()

    assert f'export VIRTUAL_ENV="{REMOTE_VIRTUAL_ENV}"' in command
    assert 'export PATH="$VIRTUAL_ENV/bin:$PATH"' in command


def test_codex_image_installs_the_complete_pinned_release_package() -> None:
    command = _codex_install_command()

    assert CODEX_RELEASE_TAG in command
    assert "codex-package-x86_64-unknown-linux-musl.tar.gz" in command
    assert "sha256sum -c -" in command
    assert "test -x /usr/local/bin/codex-code-mode-host" in command
