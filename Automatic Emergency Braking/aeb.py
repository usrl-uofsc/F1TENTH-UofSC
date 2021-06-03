#!/usr/bin/env python
import rospy
import math
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_msgs.msg import Float64
import numpy as np

class Safety(object):
    def __init__(self):
        self.TTC_limit = 1 #enter desired minimum time-to-collision
        self.view_angle = 135
        self.view_index = self.view_angle*3
        self.max_angle = 540 + self.view_index
        self.min_angle = 540 - self.view_index
        self.velx = 0
        self.ranges = []
        self.odom_sub = rospy.Subscriber('/vesc/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
        self.brake_ack = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/safety', AckermannDriveStamped, queue_size=1)
        #self.brake_bool = rospy.Publisher('/vesc/commands/motor/brake', Bool, queue_size=1)
        self.TTC_pub = rospy.Publisher('ttc', Float64, queue_size=1)
 	#self.speed = rospy.Publisher('/vesc/commands/motor/speed', Float64, queue_size=1)
        self.forward = rospy.Publisher('frwd_rng', Float64, queue_size=1)
	#self.brake_bool.publish(False)

    def odom_callback(self, odom_msg):
        self.velx = -odom_msg.twist.twist.linear.x
        self.vely = odom_msg.twist.twist.linear.y

    def scan_callback(self, scan_msg): 
        self.theta_max = scan_msg.angle_max
        self.theta_min = scan_msg.angle_min
        self.theta_step = scan_msg.angle_increment
        self.ranges = scan_msg.ranges

        self.forward_index = int(((self.theta_max - self.theta_min)/2)/self.theta_step)
        self.angles = ((np.arange(0,len(self.ranges))-self.forward_index)*self.theta_step)
        self.rel_vel = np.cos(self.angles)*self.velx
        self.rel_vel = np.maximum(self.rel_vel,float(.00000001))
        self.ranges = np.asarray(self.ranges)
	
        
        if self.velx > float(0.1):
            self.times = self.ranges/self.rel_vel
            self.min_ttc = min(self.times)
	    rospy.loginfo("min ttc: " + str(self.min_ttc))
            if (self.min_ttc <= self.TTC_limit):
                self.TTC_pub.publish(self.min_ttc)
                self.brake_command()

    def brake_command(self):
        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'closeout'
        self.ack.drive.steering_angle = 0
        self.ack.drive.speed = 0
        self.ack.header.stamp = rospy.Time.now()
	#self.speed.publish(float(0))
        #self.brake_bool.publish(True)
        self.brake_ack.publish(self.ack)
	rospy.loginfo("Brakes On!")
        #rospy.sleep(1)
	#self.brake_bool.publish(False)
	#rospy.loginfo("Brakes Off")


def main():
  rospy.init_node('safety_node')
  sn = Safety()
  rospy.spin()

if __name__ == '__main__':
    main()



    
