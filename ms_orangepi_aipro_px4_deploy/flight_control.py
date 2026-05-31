from __future__ import annotations

from pymavlink import mavutil


class PX4Controller:
    """Small MAVLink wrapper for PX4 body-frame velocity control."""

    def __init__(self, port: str = "/dev/ttyAMA1", baud: int = 921600, heartbeat_timeout: float = 10.0):
        print(f"[PX4] connecting to {port}, baud={baud}")
        self.conn = mavutil.mavlink_connection(port, baud=baud)
        msg = self.conn.recv_match(type="HEARTBEAT", blocking=True, timeout=heartbeat_timeout)
        if not msg:
            raise TimeoutError(f"no PX4 heartbeat within {heartbeat_timeout:.1f}s")
        print("[PX4] heartbeat received")

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw = 0.0

        self.conn.mav.request_data_stream_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10,
            1,
        )
        self.conn.mav.request_data_stream_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            10,
            1,
        )

    def update_telemetry(self):
        while True:
            msg = self.conn.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False)
            if not msg:
                break
            if msg.get_type() == "LOCAL_POSITION_NED":
                self.pos_x = float(msg.x)
                self.pos_y = float(msg.y)
            elif msg.get_type() == "ATTITUDE":
                self.yaw = float(msg.yaw)
        return self.pos_x, self.pos_y, self.yaw

    def send_velocity_yawrate_cmd(self, vx: float, omega: float, vy: float = 0.0, vz: float = 0.0) -> None:
        self.conn.mav.set_position_target_local_ned_send(
            0,
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            1479,
            0,
            0,
            0,
            float(vx),
            float(vy),
            float(vz),
            0,
            0,
            0,
            0,
            float(omega),
        )

    def stop_motion(self) -> None:
        self.send_velocity_yawrate_cmd(0.0, 0.0)
