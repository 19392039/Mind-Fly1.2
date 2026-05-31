from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class LidarReader:
    """Background RPLidar reader that returns 128 lidar rays in meters.

    The MindSpore policy expects ray index 0 to be forward, 32 left, 64 back,
    and 96 right. Use `angle_offset_deg` to align the physical lidar mount.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        n_rays: int = 128,
        max_distance_m: float = 10.0,
        angle_offset_deg: float = 180.0,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.n_rays = int(n_rays)
        self.max_distance_m = float(max_distance_m)
        self.angle_offset_deg = float(angle_offset_deg)
        self._scan_data = np.full((self.n_rays,), self.max_distance_m, dtype=np.float32)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lidar = None
        self._rplidar_exception = Exception

    def start(self) -> None:
        from rplidar import RPLidar, RPLidarException

        self._rplidar_exception = RPLidarException
        self._lidar = RPLidar(self.port, baudrate=self.baudrate)
        self._lidar.start_motor()
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        print(f"[Lidar] connected on {self.port}, baud={self.baudrate}")

    def get_ranges_m(self) -> np.ndarray:
        with self._lock:
            return self._scan_data.copy()

    def get_normalized(self) -> np.ndarray:
        with self._lock:
            return np.clip(self._scan_data / self.max_distance_m, 0.0, 1.0).astype(np.float32)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._lidar is not None:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass
        print("[Lidar] stopped")

    def _scan_loop(self) -> None:
        while self._running:
            try:
                for scan in self._lidar.iter_scans(max_buf_meas=500):
                    if not self._running:
                        break
                    self._process_scan(scan)
            except self._rplidar_exception:
                try:
                    self._lidar.clean_input()
                except Exception:
                    pass
                time.sleep(0.02)
            except Exception as exc:
                if self._running:
                    print(f"[Lidar] read error: {exc}")
                    time.sleep(0.2)

    def _process_scan(self, scan) -> None:
        rays = np.full((self.n_rays,), self.max_distance_m, dtype=np.float32)
        for _, angle_deg, dist_mm in scan:
            if dist_mm <= 10:
                continue
            aligned_angle = (float(angle_deg) + self.angle_offset_deg) % 360.0
            idx = int((aligned_angle / 360.0) * self.n_rays) % self.n_rays
            dist_m = min(float(dist_mm) / 1000.0, self.max_distance_m)
            if dist_m < rays[idx]:
                rays[idx] = dist_m
        with self._lock:
            self._scan_data = rays
