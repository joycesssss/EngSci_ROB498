import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger

# Optional MAVROS imports (node still runs and provides services without MAVROS installed)
try:
    from geometry_msgs.msg import PoseStamped
    from mavros_msgs.msg import State
    from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
except Exception:  # pragma: no cover
    PoseStamped = None  # type: ignore
    State = None  # type: ignore
    CommandBool = None  # type: ignore
    SetMode = None  # type: ignore
    CommandTOL = None  # type: ignore


# ----------------------------
# Global mode variable (mirrors the reference code style)
# ----------------------------
MODE = "NONE"  # NONE | LAUNCH | TEST | LAND | ABORT
LAST_TEST_ID = None  # populated if the request has a test_id field (custom srv)


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# ----------------------------
# Callback handlers (DO NOT rename)
# ----------------------------

def handle_launch():
    global MODE
    MODE = "LAUNCH"
    print('Launch Requested. Your drone should take off.')


def handle_test():
    global MODE
    MODE = "TEST"
    print('Test Requested. Your drone should perform the required tasks. Recording starts now.')


def handle_land():
    global MODE
    MODE = "LAND"
    print('Land Requested. Your drone should land.')


def handle_abort():
    global MODE
    MODE = "ABORT"
    print('Abort Requested. Your drone should land immediately due to safety considerations')


# ----------------------------
# Service callbacks (DO NOT rename)
# ----------------------------

def callback_launch(request, response):
    handle_launch()
    # If the ground server expects feedback, populate Trigger response
    if hasattr(response, "success"):
        response.success = True
        response.message = "LAUNCH accepted"
    return response


def callback_test(request, response):
    # Best-effort: capture a test identifier if a custom service is used
    global LAST_TEST_ID
    LAST_TEST_ID = getattr(request, "test_id", None)

    handle_test()
    if hasattr(response, "success"):
        response.success = True
        response.message = f"TEST accepted" + (f" (id={LAST_TEST_ID})" if LAST_TEST_ID else "")
    return response


def callback_land(request, response):
    handle_land()
    if hasattr(response, "success"):
        response.success = True
        response.message = "LAND accepted"
    return response


def callback_abort(request, response):
    handle_abort()
    if hasattr(response, "success"):
        response.success = True
        response.message = "ABORT accepted"
    return response


class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_07')

        # Required services (exact names from skeleton)
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_XX/comm/launch', callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_07/comm/test', callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_07/comm/land', callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_07/comm/abort', callback_abort)

        # Helpful aliases (some docs refer to /comm/* without the drone prefix)
        self._srv_launch_alias = self.create_service(Trigger, '/comm/launch', callback_launch)
        self._srv_test_alias = self.create_service(Trigger, '/comm/test', callback_test)
        self._srv_land_alias = self.create_service(Trigger, '/comm/land', callback_land)
        self._srv_abort_alias = self.create_service(Trigger, '/comm/abort', callback_abort)

        # ----------------------------
        # Parameters (safe defaults: manual takeoff/land allowed)
        # ----------------------------
        self.declare_parameter('setpoint_rate_hz', 20.0)          # must be > 2 Hz for PX4 offboard
        self.declare_parameter('takeoff_altitude_m', 0.50)        # adjust to 1.5 if your exercise requires
        self.declare_parameter('hover_altitude_m', 0.50)
        self.declare_parameter('altitude_tolerance_m', 0.05)
        self.declare_parameter('auto_arm', False)                 # set True if you want fully autonomous
        self.declare_parameter('auto_offboard', False)            # set True if you want code to enter OFFBOARD
        self.declare_parameter('auto_land_mode', True)            # True: AUTO.LAND, False: offboard descent
        self.declare_parameter('descent_rate_mps', 0.35)          # used if auto_land_mode=False
        self.declare_parameter('disarm_height_m', 0.12)           # disarm only when low

        self.setpoint_rate_hz = float(self.get_parameter('setpoint_rate_hz').value)
        self.takeoff_altitude_m = float(self.get_parameter('takeoff_altitude_m').value)
        self.hover_altitude_m = float(self.get_parameter('hover_altitude_m').value)
        self.altitude_tolerance_m = float(self.get_parameter('altitude_tolerance_m').value)
        self.auto_arm = bool(self.get_parameter('auto_arm').value)
        self.auto_offboard = bool(self.get_parameter('auto_offboard').value)
        self.auto_land_mode = bool(self.get_parameter('auto_land_mode').value)
        self.descent_rate_mps = float(self.get_parameter('descent_rate_mps').value)
        self.disarm_height_m = float(self.get_parameter('disarm_height_m').value)

        # ----------------------------
        # MAVROS plumbing
        # ----------------------------
        self.have_mavros = PoseStamped is not None and State is not None and CommandBool is not None

        self.current_pose: Optional[PoseStamped] = None
        self.mavros_state: Optional[State] = None

        # Mode transition tracking
        self._last_mode_seen = "NONE"

        # LAUNCH latch
        self._launch_x = 0.0
        self._launch_y = 0.0
        self._launch_yaw = 0.0
        self._launch_z_target = self.takeoff_altitude_m
        self._launch_stream_count = 0

        # TEST latch
        self._test_x = 0.0
        self._test_y = 0.0
        self._test_yaw = 0.0
        self._test_z = self.hover_altitude_m

        # LAND descent state (if doing offboard descent)
        self._land_z = self.hover_altitude_m

        # Service clients futures
        self._arm_future = None
        self._mode_future = None
        self._land_future = None

        if self.have_mavros:
            self.sub_pose = self.create_subscription(
                PoseStamped,
                '/mavros/local_position/pose',
                self._pose_cb,
                10,
            )
            self.sub_state = self.create_subscription(
                State,
                '/mavros/state',
                self._state_cb,
                10,
            )
            self.pub_setpoint = self.create_publisher(
                PoseStamped,
                '/mavros/setpoint_position/local',
                10,
            )
            self.cli_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
            self.cli_set_mode = self.create_client(SetMode, '/mavros/set_mode')
            self.cli_land = self.create_client(CommandTOL, '/mavros/cmd/land')

            self.get_logger().info('MAVROS detected: will stream setpoints on /mavros/setpoint_position/local.')
        else:
            self.get_logger().warn(
                'MAVROS not detected (mavros_msgs missing). Services will still respond, '
                'but no flight commands will be sent.'
            )

        # Timer loop (reference code used 20 Hz publishing; we keep that)
        period = 1.0 / max(self.setpoint_rate_hz, 5.0)
        self.timer = self.create_timer(period, self._tick)

    # ----------------------------
    # MAVROS callbacks
    # ----------------------------

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def _state_cb(self, msg: State) -> None:
        self.mavros_state = msg

    # ----------------------------
    # Main loop
    # ----------------------------

    def _tick(self) -> None:
        global MODE

        # Detect mode transitions (so we can latch references once)
        if MODE != self._last_mode_seen:
            self.get_logger().info(f'MODE transition: {self._last_mode_seen} -> {MODE}')
            self._on_mode_change(self._last_mode_seen, MODE)
            self._last_mode_seen = MODE

        # Nothing to do if no MAVROS
        if not self.have_mavros:
            return

        # Highest priority: ABORT
        if MODE == "ABORT":
            self._do_abort()
            return

        if MODE == "NONE":
            return

        if MODE == "LAUNCH":
            self._do_launch_or_hover()
        elif MODE == "TEST":
            self._do_test_hover()
        elif MODE == "LAND":
            self._do_land()
        else:
            self.get_logger().warn(f'Unrecognized MODE="{MODE}"; doing nothing.')

    def _on_mode_change(self, prev_mode: str, new_mode: str) -> None:
        # When we enter LAUNCH: latch XY/YAW, target Z
        if new_mode == "LAUNCH":
            self._launch_stream_count = 0
            if self.current_pose is not None:
                p = self.current_pose.pose.position
                q = self.current_pose.pose.orientation
                self._launch_x = float(p.x)
                self._launch_y = float(p.y)
                self._launch_yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            self._launch_z_target = float(self.takeoff_altitude_m)
            self._land_z = float(self.hover_altitude_m)

        # When we enter TEST: latch the reference pose for scoring
        if new_mode == "TEST":
            if self.current_pose is not None:
                p = self.current_pose.pose.position
                q = self.current_pose.pose.orientation
                self._test_x = float(p.x)
                self._test_y = float(p.y)
                self._test_yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            self._test_z = float(self.hover_altitude_m)

        # When we enter LAND: initialize land z from current
        if new_mode == "LAND":
            if self.current_pose is not None:
                self._land_z = float(self.current_pose.pose.position.z)

        # When we enter ABORT: nothing special (handled in _do_abort)

    # ----------------------------
    # Actions for each MODE
    # ----------------------------

    def _do_launch_or_hover(self) -> None:
        """Stream a hold setpoint at (launch_x, launch_y) and ascend to takeoff_altitude."""
        if self.current_pose is None:
            return

        # Setpoint streaming must be continuous (>2Hz) BEFORE OFFBOARD
        self._publish_setpoint(self._launch_x, self._launch_y, self._launch_z_target, self._launch_yaw)
        self._launch_stream_count += 1

        if self.mavros_state is None:
            return

        # Optional fully-autonomous arming/offboard
        if self.auto_arm and (not self.mavros_state.armed):
            self._request_arm(True)
            return

        # Require some setpoints before OFFBOARD
        if self.auto_offboard and (self.mavros_state.mode != 'OFFBOARD'):
            if self._launch_stream_count > int(self.setpoint_rate_hz):
                self._request_mode('OFFBOARD')
            return

        # Once at altitude, keep hovering at hover_altitude_m (still under LAUNCH until TEST arrives)
        z_now = float(self.current_pose.pose.position.z)
        if abs(z_now - self._launch_z_target) <= self.altitude_tolerance_m:
            self._launch_z_target = float(self.hover_altitude_m)

    def _do_test_hover(self) -> None:
        """Hold the pose latched at TEST time."""
        self._publish_setpoint(self._test_x, self._test_y, self._test_z, self._test_yaw)

    def _do_land(self) -> None:
        """Soft landing. Prefer AUTO.LAND, otherwise do offboard descent and disarm near ground."""
        if self.mavros_state is None or self.current_pose is None:
            return

        if self.auto_land_mode:
            # Ask FCU to land; still publish a steady setpoint as redundancy
            self._publish_setpoint(self._test_x, self._test_y, max(self.current_pose.pose.position.z, 0.2), self._test_yaw)
            self._request_land_once()
            self._request_mode('AUTO.LAND')
        else:
            # Offboard descent
            dt = 1.0 / max(self.setpoint_rate_hz, 5.0)
            self._land_z = max(self._land_z - self.descent_rate_mps * dt, 0.10)
            self._publish_setpoint(self._test_x, self._test_y, self._land_z, self._test_yaw)

        # Disarm only when low
        z_now = float(self.current_pose.pose.position.z)
        if z_now <= self.disarm_height_m and self.mavros_state.armed:
            self._request_arm(False)

    def _do_abort(self) -> None:
        """Immediate safety response: command landing and stop other behaviors."""
        if self.mavros_state is None or self.current_pose is None:
            return

        # Prefer AUTO.LAND for safety; keep publishing a conservative setpoint.
        self._publish_setpoint(
            float(self.current_pose.pose.position.x),
            float(self.current_pose.pose.position.y),
            max(float(self.current_pose.pose.position.z), 0.2),
            _yaw_from_quat(
                float(self.current_pose.pose.orientation.x),
                float(self.current_pose.pose.orientation.y),
                float(self.current_pose.pose.orientation.z),
                float(self.current_pose.pose.orientation.w),
            ),
        )
        self._request_land_once()
        self._request_mode('AUTO.LAND')

        # As a last resort, disarm only when very low
        z_now = float(self.current_pose.pose.position.z)
        if z_now <= self.disarm_height_m and self.mavros_state.armed:
            self._request_arm(False)

    # ----------------------------
    # MAVROS helpers
    # ----------------------------

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = _quat_from_yaw(float(yaw))
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        self.pub_setpoint.publish(msg)

    def _request_arm(self, arm: bool) -> None:
        if not self.cli_arm.service_is_ready():
            return
        if self._arm_future is not None and not self._arm_future.done():
            return
        req = CommandBool.Request()
        req.value = bool(arm)
        self._arm_future = self.cli_arm.call_async(req)

    def _request_mode(self, mode: str) -> None:
        if not self.cli_set_mode.service_is_ready():
            return
        if self._mode_future is not None and not self._mode_future.done():
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = str(mode)
        self._mode_future = self.cli_set_mode.call_async(req)

    def _request_land_once(self) -> None:
        if not self.cli_land.service_is_ready():
            return
        if self._land_future is not None and not self._land_future.done():
            return
        req = CommandTOL.Request()
        req.altitude = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        self._land_future = self.cli_land.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
