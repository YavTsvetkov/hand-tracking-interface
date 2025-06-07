"""
ROS 2 publisher module for sending cmd_vel commands to mobile robot.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading
import time

class RosPublisher(Node):
    """ROS 2 publisher for cmd_vel messages."""
    
    def __init__(self, node_name='hand_tracking_controller', topic_name='/cmd_vel', publish_rate=10.0):
        """
        Initialize ROS 2 publisher.
        
        Args:
            node_name: Name of the ROS 2 node
            topic_name: Topic name for cmd_vel messages
            publish_rate: Publishing rate in Hz
        """
        super().__init__(node_name)
        
        # Create publisher
        self.publisher = self.create_publisher(Twist, topic_name, 10)
        
        # Publishing parameters
        self.publish_rate = publish_rate
        self.timer_period = 1.0 / publish_rate  # seconds
        
        # Current command
        self.current_cmd = Twist()
        self.cmd_lock = threading.Lock()
        
        # Create timer for periodic publishing
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # Safety parameters
        self.safety_timeout = 1.0  # seconds
        self.last_command_time = time.time()
        
        self.get_logger().info(f'Hand tracking controller started on topic {topic_name}')
    
    def update_command(self, linear_x=0.0, angular_z=0.0):
        """
        Update the current velocity command.
        
        Args:
            linear_x: Linear velocity in x direction (m/s)
            angular_z: Angular velocity around z axis (rad/s)
        """
        with self.cmd_lock:
            self.current_cmd.linear.x = float(linear_x)
            self.current_cmd.linear.y = 0.0
            self.current_cmd.linear.z = 0.0
            self.current_cmd.angular.x = 0.0
            self.current_cmd.angular.y = 0.0
            self.current_cmd.angular.z = float(angular_z)
            self.last_command_time = time.time()
    
    def update_command_from_dict(self, cmd_dict):
        """
        Update command from dictionary format.
        
        Args:
            cmd_dict: Dictionary with 'linear_x' and 'angular_z' keys
        """
        self.update_command(
            linear_x=cmd_dict.get('linear_x', 0.0),
            angular_z=cmd_dict.get('angular_z', 0.0)
        )
    
    def timer_callback(self):
        """Timer callback for periodic publishing."""
        with self.cmd_lock:
            # Check for safety timeout
            if time.time() - self.last_command_time > self.safety_timeout:
                # Send stop command if no recent updates
                stop_cmd = Twist()
                self.publisher.publish(stop_cmd)
            else:
                # Publish current command
                self.publisher.publish(self.current_cmd)
    
    def stop_robot(self):
        """Send immediate stop command."""
        stop_cmd = Twist()
        self.publisher.publish(stop_cmd)
        self.get_logger().info('Emergency stop command sent')
    
    def get_current_command(self):
        """Get the current command values."""
        with self.cmd_lock:
            return {
                'linear_x': self.current_cmd.linear.x,
                'angular_z': self.current_cmd.angular.z,
                'timestamp': self.last_command_time
            }


class RosManager:
    """Manager class for ROS 2 operations in a separate thread."""
    
    def __init__(self, node_name='hand_tracking_controller', topic_name='/cmd_vel', publish_rate=10.0):
        """
        Initialize ROS manager.
        
        Args:
            node_name: Name of the ROS 2 node
            topic_name: Topic name for cmd_vel messages  
            publish_rate: Publishing rate in Hz
        """
        self.node_name = node_name
        self.topic_name = topic_name
        self.publish_rate = publish_rate
        
        self.publisher_node = None
        self.ros_thread = None
        self.running = False
        
    def start(self):
        """Start ROS 2 in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.ros_thread = threading.Thread(target=self._ros_thread_worker)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        
        # Wait for node to be ready
        time.sleep(0.5)
        
    def stop(self):
        """Stop ROS 2 operations."""
        if not self.running:
            return
        
        self.running = False
        
        if self.publisher_node:
            self.publisher_node.stop_robot()
            self.publisher_node.destroy_node()
        
        if self.ros_thread and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=2.0)
    
    def _ros_thread_worker(self):
        """Worker function for ROS 2 thread."""
        try:
            # Initialize ROS 2
            rclpy.init()
            
            # Create publisher node
            self.publisher_node = RosPublisher(
                node_name=self.node_name,
                topic_name=self.topic_name,
                publish_rate=self.publish_rate
            )
            
            # Spin the node
            while self.running and rclpy.ok():
                rclpy.spin_once(self.publisher_node, timeout_sec=0.1)
                
        except Exception as e:
            print(f"[ERROR] ROS thread error: {e}")
        finally:
            if self.publisher_node:
                self.publisher_node.destroy_node()
            rclpy.shutdown()
    
    def update_command(self, linear_x=0.0, angular_z=0.0):
        """Update velocity command."""
        if self.publisher_node:
            self.publisher_node.update_command(linear_x, angular_z)
    
    def update_command_from_dict(self, cmd_dict):
        """Update command from dictionary."""
        if self.publisher_node:
            self.publisher_node.update_command_from_dict(cmd_dict)
    
    def stop_robot(self):
        """Send stop command."""
        if self.publisher_node:
            self.publisher_node.stop_robot()
    
    def get_current_command(self):
        """Get current command values."""
        if self.publisher_node:
            return self.publisher_node.get_current_command()
        return {'linear_x': 0.0, 'angular_z': 0.0, 'timestamp': 0.0}
