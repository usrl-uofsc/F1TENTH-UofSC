#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

#PID CONTROL PARAMS
kp = 1
kd = 0.5
ki = 0.125
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0
angle_a = 10
#WALL FOLLOW PARAMS
ANGLE_RANGE = 270 # Hokuyo 10LX has 270 degrees scan
DESIRED_DISTANCE_RIGHT = 0.9 # meters
DESIRED_DISTANCE_LEFT = 0.75
VELOCITY = 2.00 # meters per second
CAR_LENGTH = 0.50 # Traxxas Rally is 20 inches or 0.5 meters
LOOKAHEAD = 0.5

class WallFollow:
    """ Implement Wall Following on the car
    """
    def __init__(self):
        #Topics & Subs, Pubs
        self.start_time = rospy.get_time()*1000
        self.old_time = self.start_time
        lidarscan_topic = '/scan'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/input/navigation'

        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback) #Subscribe to LIDAR
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1) #Publish to drive

    def getRange(self, data, angle):
            rng = data.ranges[angle]
            while (np.isnan(rng)) or (np.isinf(rng)):
                rng = data.ranges[angle]
            return rng

        # data: single message from topic /scan
        # angle: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view
        #make sure to take care of nans etc.
        #TODO: implement

    def pid_control(self, error, velocity):
        global integral
        global prev_error
        global kp
        global ki
        global kd
        self.current_time = rospy.get_time()*1000
        dt = self.current_time - self.old_time
        self.old_time = self.current_time  
        tt = self.current_time - self.start_time
        #angle = 0.0
        angle = float(kp*error + (ki*integral/tt) + kd*((error - prev_error)/dt)) #Use kp, ki & kd to implement a PID controller for 
        integral = (integral + error)
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        if 0 <= abs(angle) < 10:
            velocity = -1.0
        elif 10 <= abs(angle) < 20:
            velocity = -1.0
        else:
            velocity = -0.5
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)
        rospy.loginfo(angle)
        prev_error = error

    def followLeft(self, data, leftDist):
        global angle_a
        b_inc = 810
        a_inc = 810 - 3 * angle_a
         #signifies number of increments from start angle to angle that is the the car's right (Do Not Change)
        inc = data.angle_increment
        start_ang = data.angle_min
        L = LOOKAHEAD
        self.a_ang = start_ang+(inc * b_inc)
        self.b_ang = start_ang+(inc * a_inc)
        self.theta = float(-(self.b_ang-self.a_ang))
        self.b = self.getRange(data, b_inc)
        self.a = self.getRange(data, a_inc)
        self.alpha = math.atan(((self.a * math.cos(self.theta))-self.b)/(self.a*math.sin(self.theta)))
        D_t = self.b * math.cos(self.alpha)
        D_t1 = D_t + L*math.sin(self.alpha)

        self.error = (D_t1 - leftDist)
        
        return self.error 

    def lidar_callback(self, data):
        """ 
        """
        error = self.followLeft(data, DESIRED_DISTANCE_LEFT) #replace with error returned by followLeft (done)
        #send error to pid_control
        self.pid_control(error, VELOCITY)

def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    wf = WallFollow()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
