"""Continuous live capture with independently sampled MP4 recording."""

from __future__ import annotations

import threading
import time
import json
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..hardware.camera import CameraSet, CapturedImage
from .h264 import H264Mp4Writer


class RunVideo:
    """Record one MP4 per camera while keeping inference capture independent."""

    def __init__(self, cameras: CameraSet, directory: Path, fps: int = 10) -> None:
        if fps <= 0:
            raise ValueError("录像 fps 必须大于零")
        self.cameras, self.fps = cameras, fps
        names = [camera.name for camera in cameras.cameras]
        if not names or len(set(names)) != len(names) or any(
            not name or Path(name).name != name for name in names
        ):
            raise ValueError("相机名称须唯一且不能包含路径")
        with ExitStack() as files:
            self.writers = {
                camera.name: files.enter_context(H264Mp4Writer(
                    directory / f"{camera.name}.mp4", camera.width, camera.height, fps,
                ))
                for camera in cameras.cameras
            }
            self.timestamps = files.enter_context((directory / "video-frames.jsonl").open("x", encoding="utf-8", buffering=1))
            self._files = files.pop_all()
        self._latest: dict[str, CapturedImage] | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._capture_error: Exception | None = None
        self._record_error: Exception | None = None
        self._capture_frames = 0
        self._video_frames = 0
        self._started_at = time.time()
        self._ended_at: float | None = None
        self._capture_thread = threading.Thread(target=self._capture, name="robot-cameras", daemon=True)
        self._record_thread = threading.Thread(target=self._record, name="robot-video", daemon=True)
        self._capture_thread.start()
        self._record_thread.start()

    def snapshot(
        self, after: float | None = None, timeout: float | None = None,
    ) -> dict[str, CapturedImage]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_capture_error()
                if self._record_error is not None:
                    raise RuntimeError(f"录像编码失败: {self._record_error}") from self._record_error
                if self._latest and (after is None or min(x.captured_at for x in self._latest.values()) > after):
                    return dict(self._latest)
                if self._stop.is_set():
                    raise RuntimeError("相机采集已停止")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("等待新的相机画面超时")
                self._condition.wait(remaining)

    @property
    def details(self) -> dict[str, Any]:
        return {
            "state": "stopped" if self._stop.is_set() else "recording",
            "videos": {
                name: {
                    "path": writer.path.name,
                    "width": writer.width,
                    "height": writer.height,
                    "frames": writer.frames,
                }
                for name, writer in self.writers.items()
            },
            "codec": "H.264",
            "container": "MP4",
            "fps": self.fps,
            "frames": self._video_frames,
            "captured_batches": self._capture_frames,
            "started_at_s": self._started_at,
            "ended_at_s": self._ended_at,
            "frame_timestamps": "video-frames.jsonl",
        }

    def stop(self) -> dict[str, Any]:
        if not self._stop.is_set():
            # Called after return_home(): retain a fresh view of the final pose.
            try:
                self.snapshot(after=time.time(), timeout=2.0)
            except Exception as exc:
                self._capture_error = exc
            self._ended_at = time.time()
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._capture_thread.join(timeout=5)
        self._record_thread.join(timeout=5)
        if self._capture_thread.is_alive() or self._record_thread.is_alive():
            raise TimeoutError("Camera/video thread did not stop within 5 seconds")
        # Closing finalizes each MP4; report an earlier capture/record error first.
        try:
            self._files.close()
        except Exception as exc:
            self._raise_capture_error()
            if self._record_error is None:
                raise RuntimeError(f"录像编码失败: {exc}") from exc
        self._raise_capture_error()
        if self._record_error is not None:
            raise RuntimeError(f"录像编码失败: {self._record_error}") from self._record_error
        return self.details

    def _capture(self) -> None:
        try:
            while not self._stop.is_set():
                images = self.cameras.capture(stop=self._stop)
                with self._condition:
                    self._latest = images
                    self._capture_frames += 1
                    self._condition.notify_all()
        except Exception as exc:
            if not self._stop.is_set():
                self._capture_error = exc
            with self._condition:
                self._condition.notify_all()

    def _record(self) -> None:
        period = 1.0 / self.fps
        try:
            self.snapshot()
            deadline = time.monotonic()
            while not self._stop.is_set():
                images = self.snapshot()
                self._write_frames(images)
                # Encoding time is part of the period, not an extra delay.
                deadline += period
                self._stop.wait(max(0.0, deadline - time.monotonic()))
            # stop() first requests a fresh batch, so this includes the home pose.
            self._write_frames(self.snapshot())
        except Exception as exc:
            self._record_error = exc
            with self._condition:
                self._condition.notify_all()

    def _write_frames(self, images: dict[str, CapturedImage]) -> None:
        # Each camera keeps its own resolution; encoding and I/O hold no capture lock.
        for name, writer in self.writers.items():
            image = images[name]
            if image.rgb_data is not None:
                rgb = np.frombuffer(image.rgb_data, dtype=np.uint8).reshape(
                    image.height, image.width, 3,
                )
            elif image.mime_type == "image/jpeg":
                # Camera MJPEG payloads are decoded once for the H.264 encoder.
                with Image.open(BytesIO(image.data)) as decoded:
                    rgb = np.asarray(decoded.convert("RGB"))
            else:
                raise RuntimeError(f"相机 {name} 缺少可录制的 RGB/JPEG 画面")
            writer.append(rgb)
            self.timestamps.write(json.dumps({
                "camera": name, "frame_index": self._video_frames,
                "captured_at_s": image.captured_at,
                "source_timestamp_s": image.source_timestamp_s, "source_clock": image.source_clock,
                "recorded_at_s": time.time(),
                "width": writer.width, "height": writer.height, "fps": self.fps,
            }) + "\n")
        self._video_frames += 1

    def _raise_capture_error(self) -> None:
        if self._capture_error is not None:
            raise RuntimeError(f"相机采集失败: {self._capture_error}") from self._capture_error
