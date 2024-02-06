import rclpy

from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory , JointTrajectoryPoint


# Define a class named TrajectoryPublisher that inherits from the Node class
class TrajectoryPublisher(Node):

    # Constructor method
    def __init__(self):
        # Call the constructor of the parent class (Node) using super()
        super().__init__('robot_joint_trajectory_publisher')
        
        # # Set the timer period to 1 second
        timer_period = 1

        # # Create a timer and associate it with the timer_callback method
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # # Create a publisher for the JointTrajectory message on the specified topic
        self.trajectory_publisher = self.create_publisher(JointTrajectory,"/joint_trajectory_controller/joint_trajectory", 10)


        # self.position =        


    # # Callback method for the timer
    def timer_callback(self):

        self.get_logger().info('Hello!!!')

    #     # Create a list containing a single value (-0.7) representing goal positions
        goal_positions = [-0.7,]
        
    #     # Create a JointTrajectoryPoint message and set its positions and time_from_start attributes
        point_msg = JointTrajectoryPoint()
        point_msg.positions = goal_positions
        
        point_msg.time_from_start = Duration(sec=2)


    #     # Create a list of joint names containing a single value ('joint1')
        joints = ['wheel1_joint']

    #     # Create a JointTrajectory message and set its joint_names attribute and append the point_msg to its points attribute
        my_trajectory_msg = JointTrajectory()
        my_trajectory_msg.joint_names = joints
        my_trajectory_msg.points.append(point_msg)
        
    #     # Publish the JointTrajectory message
        self.trajectory_publisher.publish(my_trajectory_msg)


# Main function
def main(args=None):

    # Initialize the ROS Client Library (rclpy)
    rclpy.init(args=args)

    # Create an instance of the TrajectoryPublisher class
    joint_trajectory_object = TrajectoryPublisher()

    # Spin (keep the node running) until it is shutdown
    rclpy.spin(joint_trajectory_object)
    
    # Destroy the node
    joint_trajectory_object.destroy_node()

    # Shutdown the ROS Client Library
    rclpy.shutdown()



if __name__ == '__main__':
    main()
