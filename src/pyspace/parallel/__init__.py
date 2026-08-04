"""Opt-in resource measurement with no import-time hardware probe."""

from .resources import ResourceSnapshot, current_process_resources

__all__ = ["ResourceSnapshot", "current_process_resources"]
