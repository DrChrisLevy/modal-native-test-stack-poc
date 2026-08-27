"""Errors raised by the Modal execution harness."""


class ModalNativeTestStackError(RuntimeError):
    """A user-facing failure with enough context to fix the remote run."""
