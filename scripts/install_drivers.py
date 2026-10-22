#!/usr/bin/env python3
"""Install optional hardware backends for the public package."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("arx", "yam", "ur5", "realsense"), nargs="+")
    args = parser.parse_args()
    packages = {
        "arx": ["arx5-interface==0.1.3"],
        "yam": ["i2rt @ git+https://github.com/i2rt-robotics/i2rt.git@ac096928d6899ddf852a71c5e8fbaa6055cd9745"],
        "ur5": ["ur-rtde>=1.5"],
        "realsense": ["pyrealsense2"],
    }
    selected = []
    for backend in args.backend:
        for package in packages[backend]:
            if package not in selected:
                selected.append(package)
    subprocess.run([sys.executable, "-m", "pip", "install", *selected], check=True)


if __name__ == "__main__":
    main()
