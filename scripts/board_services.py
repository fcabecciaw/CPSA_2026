"""Refuse to open the F200 while the demo services are running.

Shared by the capture scripts that run **on the board**. On 2026-11-11 the board
froze when an IR capture ran next to the demo services, so every script that
opens the camera first checks ``systemctl is-active`` for the units below and
stops with an actionable message unless ``--allow-services`` is given.

The check fails closed: if ``systemctl`` cannot be run, or its output cannot be
mapped back to the requested units, :class:`ServiceCheckError` is raised rather
than letting a capture race the demo.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

__all__ = ["SERVICES", "ServiceCheckError", "active_services", "services_refusal"]

SERVICES = ("cpsa-supervisor", "cpsa-dpu")
"""Board units that must be stopped before the camera is opened."""

#: ``systemctl is-active`` states that mean the unit is (still) running.
BUSY_STATES = frozenset({"active", "activating", "reloading", "deactivating"})

STOP_COMMAND = "sudo systemctl stop cpsa-supervisor cpsa-dpu"
"""What the operator runs before retrying a capture."""


class ServiceCheckError(RuntimeError):
    """The state of the demo services could not be determined."""


def active_services(
    units: Sequence[str] = SERVICES,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, ...]:
    """Return the units of ``units`` that ``systemctl is-active`` reports busy.

    Args:
        units: systemd unit names to query in one call.
        run: Test seam; defaults to :func:`subprocess.run`.

    Raises:
        ServiceCheckError: If ``systemctl`` cannot be run or its output cannot be
            mapped to the units.  The check fails closed: an unknown state must
            not let a capture race the demo services.
    """

    try:
        completed = run(
            ["systemctl", "is-active", *units],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ServiceCheckError(f"cannot run systemctl: {exc}") from exc
    states = completed.stdout.split()
    if len(states) != len(units):
        raise ServiceCheckError(
            f"systemctl is-active printed {len(states)} states for {len(units)} units: "
            f"{completed.stdout.strip()!r}"
        )
    return tuple(unit for unit, state in zip(units, states, strict=True) if state in BUSY_STATES)


def services_refusal(busy: Sequence[str], *, action: str = "open the camera") -> str:
    """Return the multi-line refusal printed when ``busy`` units are active."""

    return (
        f"refusing to {action}: {', '.join(busy)} active. The board froze on 2026-11-11 when\n"
        "a capture ran alongside the demo services. Stop them first:\n"
        f"    {STOP_COMMAND}\n"
        "then rerun this command (--allow-services skips this check)."
    )
