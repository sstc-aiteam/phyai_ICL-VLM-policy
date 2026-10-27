"""Read a UR arm's factory kinematic calibration from the controller (read-only).

    python -m gpt_policy.hardware.ur5_kinematics 192.168.50.51

Prints the ``robot.kinematics_calibration`` block for the machine profile. It listens on the
primary interface (port 30001) and sends nothing, so the arm and the controller state are untouched.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct

PRIMARY_PORT = 30001
ROBOT_STATE_PACKAGE = 16
KINEMATICS_SUBPACKAGE = 5
MAX_PACKAGES = 60


def _receive(sock: socket.socket, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("The robot closed the primary interface")
        data += chunk
    return data


def read_kinematics(ip: str, timeout_s: float = 5.0) -> dict:
    with socket.create_connection((ip, PRIMARY_PORT), timeout=timeout_s) as sock:
        for _ in range(MAX_PACKAGES):
            size, kind = struct.unpack(">IB", _receive(sock, 5))
            body = _receive(sock, size - 5)
            if kind != ROBOT_STATE_PACKAGE:
                continue
            offset = 0
            while offset + 5 <= len(body):
                sub_size, sub_kind = struct.unpack(">IB", body[offset:offset + 5])
                if sub_kind == KINEMATICS_SUBPACKAGE:
                    values = struct.unpack(">24d", body[offset + 5 + 24:offset + 5 + 24 + 192])
                    return {name: list(values[i * 6:(i + 1) * 6])
                            for i, name in enumerate(("dh_theta", "dh_a", "dh_d", "dh_alpha"))}
                if sub_size < 5:
                    break
                offset += sub_size
    raise RuntimeError("The robot did not send its kinematics info")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ip")
    args = parser.parse_args(argv)
    print(json.dumps({"kinematics_calibration": read_kinematics(args.ip)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
