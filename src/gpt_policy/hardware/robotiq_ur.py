"""Robotiq 2F gripper on a UR tool connector, driven through RTDE custom scripts.

This matches the working setup in ``phyai_vla-ur7``: the vendor ``rq_*`` URScript
preamble is sent with ``sendCustomScriptFunction``. That path is open loop
(the script cannot return the finger position to the host), so ``opening`` is the
last commanded value, not a measurement. The preamble itself is the ur_rtde
example script; it is loaded from a file you point at rather than bundled here.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

RAW_MAX = 255  # Robotiq position register: 0 fully open, 255 fully closed.


def load_preamble(path: str | Path | None) -> str:
    """Read ``ROBOTIQ_PREAMBLE`` from a Python file, or from ``ur7e_recorder`` if installed."""
    if path is None:
        try:
            from ur7e_recorder.gripper.robotiq_preamble import ROBOTIQ_PREAMBLE
        except ImportError as exc:
            raise RuntimeError(
                "Set robot.robotiq_preamble to a Python file defining ROBOTIQ_PREAMBLE "
                "(the ur_rtde Robotiq preamble), or install ur7e_recorder.") from exc
        return ROBOTIQ_PREAMBLE
    file = Path(path).expanduser()
    spec = importlib.util.spec_from_file_location("robotiq_preamble", file)
    if spec is None or spec.loader is None or not file.is_file():
        raise FileNotFoundError(f"Robotiq preamble file not found: {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preamble = getattr(module, "ROBOTIQ_PREAMBLE", None)
    if not isinstance(preamble, str) or "rq_move_and_wait" not in preamble:
        raise ValueError(f"{file} does not define a usable ROBOTIQ_PREAMBLE")
    return preamble


def opening_to_raw(opening: float) -> int:
    """Normalized opening (0 closed, 1 open) to the Robotiq register (0 open, 255 closed)."""
    if not 0.0 <= opening <= 1.0:
        raise ValueError("gripper opening must be in [0, 1]")
    return int(round((1.0 - opening) * RAW_MAX))


class RobotiqUR:
    """Normalized-opening wrapper; ``link`` provides ``run_script(name, text)``."""

    def __init__(self, link, preamble: str, *, speed_percent=50, force_percent=50,
                 initial_opening=1.0, activate=True, activation_wait_s=5.0, settle_s=0.2):
        for name, value in (("speed_percent", speed_percent), ("force_percent", force_percent)):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        self.link, self.preamble, self.settle_s = link, preamble, float(settle_s)
        self.opening = float(initial_opening)
        if activate:
            self._call("ACTIVATE", "rq_activate()")
            time.sleep(activation_wait_s)  # The script reports no completion.
            self._call("SET_SPEED", f"rq_set_speed_norm({int(speed_percent)})")
            self._call("SET_FORCE", f"rq_set_force_norm({int(force_percent)})")

    def _call(self, name: str, body: str):
        return self.link.run_script("ROBOTIQ_" + name, self.preamble + body)

    def move(self, opening: float) -> dict:
        raw = opening_to_raw(opening)
        started = time.monotonic()
        # "and_wait" blocks the controller script until the fingers stop or grip.
        accepted = self._call("MOVE", f"rq_move_and_wait({raw})")
        time.sleep(self.settle_s)
        self.opening = float(opening)
        return {"raw_position": raw, "script_accepted": bool(accepted),
                "duration_s": time.monotonic() - started,
                "feedback": "open_loop_last_command"}


class NoGripper:
    """Bare arm: the opening is only a bookkeeping value."""

    def __init__(self, initial_opening=1.0):
        self.opening = float(initial_opening)

    def move(self, opening: float) -> dict:
        opening_to_raw(opening)
        self.opening = float(opening)
        return {"feedback": "no_gripper"}
