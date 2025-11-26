# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------
# Modifications:
#   - Added time-based tapered acceleration and deceleration at the
#     motor layer so throttle and rotation ramp smoothly instead of
#     stepping instantly to the commanded value.
#   - Updated by Alex Anderson, University of Portland, 2025.
# ----------------------------------------------------------------------

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pysabertooth import Sabertooth
from serial import SerialException

'''
    This node controls the driving of the robot via two Sabertooth motor controllers.
    It now includes *tapered acceleration and deceleration* at the motor layer.

    High-level nodes (joystick/autonomy) can still command /cmd/velocity,
    but the actual wheel speeds will ramp over a short time window instead
    of jumping instantly. This reduces jerk and improves traction.

    Subscriptions:
    /cmd/velocity - Twist
        linear.x : desired forward/backward "throttle" (-100..100)
        angular.z: desired rotation rate (we treat +/-50 as full turn)
'''


class ReconnectableSaber:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, address=128, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.timeout = timeout
        self.saber = None
        self._connect()

    def _connect(self):
        try:
            self.saber = Sabertooth(
                self.port,
                baudrate=self.baudrate,
                address=self.address,
                timeout=self.timeout,
            )
            print("[ReconnectableSaber] Connected to Sabertooth on", self.port)
        except Exception as e:
            print(f"[ReconnectableSaber] Failed to connect on {self.port}: {e}")
            self.saber = None

    def drive(self, motor, speed):
        """
        speed is expected to be an int in roughly -100..100.
        """
        if not self.saber:
            self._connect()
            if not self.saber:
                print("[ReconnectableSaber] Still not connected. Skipping command.")
                return

        try:
            self.saber.drive(motor, int(speed))
        except SerialException as e:
            print(f"[ReconnectableSaber] SerialException: {e}. Reconnecting...")
            self._connect()
        except Exception as e:
            print(f"[ReconnectableSaber] Unexpected error: {e}")

    def stop(self):
        if not self.saber:
            self._connect()
            if not self.saber:
                print("[ReconnectableSaber] Still not connected. Skipping stop().")
                return

        try:
            self.saber.stop()
        except SerialException as e:
            print(f"[ReconnectableSaber] SerialException on stop(): {e}. Reconnecting...")
            self._connect()
        except Exception as e:
            print(f"[ReconnectableSaber] Unexpected error on stop(): {e}")


# left wheels
saber = ReconnectableSaber('/dev/ttyACM0', baudrate=9600, address=128, timeout=0.1)
# right wheels
saber2 = ReconnectableSaber('/dev/ttyACM1', baudrate=9600, address=128, timeout=0.1)

DIR = 1.0          # 1 is the correct direction, set to -1 for backwards
TURN_SPEED = 50.0  # turn speed when turning in place

# --- Tapered acceleration / deceleration parameters -------------------------

MAX_THROTTLE = 100.0     # expected |linear.x| max coming from joystick/autonomy
MAX_ROT_SPEED = 50.0     # expected |angular.z| max

ACCEL_TIME_THROTTLE = 2.0   # seconds to go from 0 -> full throttle (update these 4 values for more or less tapered speed
DECEL_TIME_THROTTLE = 1.5   # seconds to go from full -> 0

ACCEL_TIME_ROT = 1.0        # seconds to go 0 -> full turn rate
DECEL_TIME_ROT = 1.0        # seconds to go full turn -> 0

UPDATE_PERIOD = 0.02        # 50 Hz update rate for ramping (seconds)


def _ramp_value(current, target, max_step_accel, max_step_decel):
    """
    Move 'current' toward 'target' by no more than max_step_* per update.

    - If we are speeding up (|target| > |current|) we use max_step_accel.
    - If we are slowing down (|target| <= |current|) we use max_step_decel.
    """
    delta = target - current
    # if abs dif is small then values are equal
    if abs(delta) < 0.000001:
        return current

    speeding_up = abs(target) > abs(current)
    max_step = max_step_accel if speeding_up else max_step_decel

    if abs(delta) <= max_step:
        return target

    return current + math.copysign(max_step, delta)


class DriveMotors(Node):

    def __init__(self):
        super().__init__('drive_motors')

        # Latest commanded values from /cmd/velocity (high-level)
        self.target_throttle = 0.0    # desired forward/back [-100..100]
        self.target_rotation = 0.0    # desired rotation, typically -50..50

        # Smoothed values that we actually send to Sabertooths
        self.current_throttle = 0.0
        self.current_rotation = 0.0

        # Subscribe to high-level velocity commands
        self.subscription = self.create_subscription(
            Twist,
            'cmd/velocity',
            self.listener_callback,
            10
        )

        # Timer to apply ramping and talk to the Sabertooths at steady rate
        self.timer = self.create_timer(UPDATE_PERIOD, self.update_motors)

        self.last_update_time = self.get_clock().now()

        self.get_logger().info('DriveMotors initialized with tapered accel/decel.')

    # Called when new /cmd/velocity arrives
    def listener_callback(self, msg: Twist):
        # Clamp incoming commands to our expected ranges
        throttle = float(msg.linear.x)
        rotation = float(msg.angular.z)

        if throttle > MAX_THROTTLE:
            throttle = MAX_THROTTLE
        elif throttle < -MAX_THROTTLE:
            throttle = -MAX_THROTTLE

        if rotation > MAX_ROT_SPEED:
            rotation = MAX_ROT_SPEED
        elif rotation < -MAX_ROT_SPEED:
            rotation = -MAX_ROT_SPEED

        self.target_throttle = throttle
        self.target_rotation = rotation

    # Runs at 50 Hz, smoothly moves current_* toward target_* and sends motor cmds
    def update_motors(self):
        now = self.get_clock().now()
        dt = (now.nanoseconds - self.last_update_time.nanoseconds) / 1e9
        if dt <= 0.0:
            dt = UPDATE_PERIOD
        self.last_update_time = now

        # Compute max change per update step based on desired ramp times
        # (These scale with dt so it behaves the same at different loop rates)
        max_step_accel_throttle = (MAX_THROTTLE * dt) / ACCEL_TIME_THROTTLE
        max_step_decel_throttle = (MAX_THROTTLE * dt) / DECEL_TIME_THROTTLE

        max_step_accel_rot = (MAX_ROT_SPEED * dt) / ACCEL_TIME_ROT
        max_step_decel_rot = (MAX_ROT_SPEED * dt) / DECEL_TIME_ROT

        # Smoothly move toward targets
        self.current_throttle = _ramp_value(
            self.current_throttle,
            self.target_throttle,
            max_step_accel_throttle,
            max_step_decel_throttle
        )

        self.current_rotation = _ramp_value(
            self.current_rotation,
            self.target_rotation,
            max_step_accel_rot,
            max_step_decel_rot
        )

        throttle = self.current_throttle
        rotation = self.current_rotation

        # Decide motor outputs using the same logic as original code,
        # but using *smoothed* throttle/rotation instead of raw commands.
        if (throttle != 0.0 and rotation == 0.0):
            # Straight forward/back
            left_1  = throttle * DIR
            left_2  = -throttle * DIR
            right_1 = throttle * DIR
            right_2 = -throttle * DIR

            saber.drive(1, left_1)
            saber.drive(2, left_2)
            saber2.drive(1, right_1)
            saber2.drive(2, right_2)

        elif (throttle != 0.0 and rotation != 0.0):
            # Curved motion (arc turn)
            t1 = throttle
            t2 = throttle

            if rotation < 0.0:   # left turn
                t2 = t2 * 0.5    # slow down right side
            elif rotation > 0.0: # right turn
                t1 = t1 * 0.5    # slow down left side

            left_1  = t1 * DIR
            left_2  = -t1 * DIR
            right_1 = t2 * DIR
            right_2 = -t2 * DIR

            saber.drive(1, left_1)
            saber.drive(2, left_2)
            saber2.drive(1, right_1)
            saber2.drive(2, right_2)

        elif (throttle == 0.0 and rotation != 0.0):
            # Turn in place, scaled by sign of rotation
            if rotation < 0.0:  # left turn
                saber2.drive(1, -TURN_SPEED * DIR)
                saber2.drive(2, TURN_SPEED * DIR)
                saber.drive(1, TURN_SPEED * DIR)
                saber.drive(2, -TURN_SPEED * DIR)
            elif rotation > 0.0:  # right turn
                saber.drive(1, -TURN_SPEED * DIR)
                saber.drive(2, TURN_SPEED * DIR)
                saber2.drive(1, TURN_SPEED * DIR)
                saber2.drive(2, -TURN_SPEED * DIR)
        else:
            # Both commands effectively zero -> stop motors
            saber.stop()
            saber2.stop()


def main(args=None):
    rclpy.init(args=args)

    node = DriveMotors()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down drive_motors.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
