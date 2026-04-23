import math
from typing import Dict

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Range


class EpuckBraitenbergNode(Node):
    def __init__(self):
        super().__init__('epuck_braitenberg')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sensor_values: Dict[int, float] = {i: 0.0 for i in range(8)}
        self.last_ranges: Dict[int, float] = {i: math.inf for i in range(8)}

        self.forward_speed = self.declare_parameter('forward_speed', 0.05).value
        self.turn_gain = self.declare_parameter('turn_gain', 6.0).value
        self.front_threshold = self.declare_parameter('front_threshold', 0.04).value
        self.avoid_gain = self.declare_parameter('avoid_gain', 1.0).value
        self.max_turn = self.declare_parameter('max_turn', 2.5).value
        self.timer_period = self.declare_parameter('timer_period', 0.05).value

        for i in range(8):
            self.create_subscription(
                Range,
                f'/ps{i}',
                lambda msg, idx=i: self.range_callback(idx, msg),
                10,
            )

        self.timer = self.create_timer(self.timer_period, self.control_step)
        self.log_counter = 0
        self.get_logger().info('epuck_braitenberg node started')

    def range_callback(self, idx: int, msg: Range):
        self.last_ranges[idx] = msg.range
        max_range = msg.max_range if msg.max_range > 0.0 else 0.07
        closeness = 1.0 - min(max(msg.range / max_range, 0.0), 1.0)
        self.sensor_values[idx] = max(0.0, min(1.0, closeness))

    def control_step(self):
        s = self.sensor_values
        left_side = (s[5] + s[6] + s[7]) / 3.0
        right_side = (s[0] + s[1] + s[2]) / 3.0
        front = max(s[0], s[1], s[6], s[7])

        twist = Twist()
        twist.linear.x = self.forward_speed * max(0.0, 1.0 - self.avoid_gain * front)
        twist.angular.z = self.turn_gain * (right_side - left_side)

        if front > (1.0 - self.front_threshold / 0.0666):
            twist.linear.x *= 0.2
            twist.angular.z += self.max_turn if right_side <= left_side else -self.max_turn

        twist.angular.z = max(-self.max_turn, min(self.max_turn, twist.angular.z))
        self.cmd_pub.publish(twist)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            self.get_logger().info(
                f'front={front:.2f} left={left_side:.2f} right={right_side:.2f} '
                f'cmd=({twist.linear.x:.3f} m/s, {twist.angular.z:.3f} rad/s)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = EpuckBraitenbergNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
