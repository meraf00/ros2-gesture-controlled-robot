import rclpy

from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist

class RobotControllerNode(Node):
    
    def __init__(self):        
        super().__init__('robot_controller')
        self.get_logger().info('Robot controller started')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)                        
        
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.send_velocity_cmd)

    
    def send_velocity_cmd(self):
        msg = Twist()

        msg.linear.x = -30.0                

        print(msg)

        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = RobotControllerNode()
    
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
