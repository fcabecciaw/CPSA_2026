"""Composition root for scripted RGB-D and wrist-IMU session recording."""

from __future__ import annotations

from cpsa.recorder.protocol import (
    Protocol,
    ProtocolClass,
    ProtocolError,
    TrialSpec,
    load,
    plan_trials,
    protocol_sha256,
)
from cpsa.recorder.session import RunResult, SessionController

__all__ = [
    "Protocol",
    "ProtocolClass",
    "ProtocolError",
    "RunResult",
    "SessionController",
    "TrialSpec",
    "load",
    "plan_trials",
    "protocol_sha256",
]
