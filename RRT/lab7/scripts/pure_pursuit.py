#!/usr/bin/env python
import rospy
import sys
import os

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import numpy as np
import math

# TODO: import ROS msg types and libraries
LOOKAHEAD = 1.2
Waypoint_CSV_File_Path = '/home/zach/catkin_ws/src/lab6/waypoints/levine-waypoints.csv'
Odom_Topic = rospy.get_param("/pose_topic")
Car_Length = 0.32

class PurePursuit(object):
    def __init__(self):
        global Waypoint_CSV_File_Path
        global LOOKAHEAD
        global Car_Length

        self.L = LOOKAHEAD
        self.car_length = Car_Length

        np.set_printoptions(threshold=sys.maxsize)
        self.iter = 0

        rospy.Subscriber("/pure_pursuit", Float32MultiArray, self.pose_callback, queue_size=1) #TODO
    
        self.drive_pub = rospy.Publisher('drive', AckermannDriveStamped, queue_size=1)

    def FindNavIndex(self, distances, L):
        min_index = np.argmin(distances)
        
        differences = np.subtract(distances,L)
        next_differences = np.roll(differences, -1)

        i = min_index
        while 1:
            if i > (len(differences)-1):
                i = 0

            if np.sign(differences[i]) != np.sign(next_differences[i]):
                return i
            else:
                i += 1

    def FindNavPoint(self, goal_index, magnitudes, waypoints, L):
        if goal_index == len(waypoints)-1:
            next_index = 0
        else:
            next_index = goal_index + 1

        mi = 0 
        m1 = magnitudes[goal_index] - L
        m2 = magnitudes[next_index] - L
        x1 = waypoints[goal_index][0]
        x2 = waypoints[next_index][0]
        y1 = waypoints[goal_index][1]
        y2 = waypoints[next_index][1]

        xi = np.interp(mi, [m1,m2], [x1, x2])
        yi = np.interp(mi, [m1,m2], [y1, y2])

        goal_point = np.asarray([xi,yi])
        return goal_point

    def pose_callback(self, wp_msg):
        print(self.iter)
        self.iter += 1
        height = wp_msg.layout.dim[0].size
        width = wp_msg.layout.dim[1].size
        data = np.asarray(wp_msg.data)

        self.Waypoints_Master = np.reshape(data, (height, width))
        
        L = self.L

        car_length = self.car_length
        waypoints = self.Waypoints_Master[0:, 1:]

        car_point = self.Waypoints_Master[-1, 1:]
        angle_z = self.Waypoints_Master[0, 0]

        magnitudes = np.asarray([np.linalg.norm(waypoint - car_point) for waypoint in waypoints])

        goal_index = self.FindGoalIndex(magnitudes, L)

        goal_point = self.FindGoalPoint(goal_index, magnitudes, waypoints, L)
        
        x = (goal_point[0] - car_point[0])*math.cos(angle_z) + (goal_point[1] - car_point[1])*math.sin(angle_z)
        y = -(goal_point[0] - car_point[0])*math.sin(angle_z) + (goal_point[1] - car_point[1])*math.cos(angle_z)

        goal_for_car = np.asarray([x, y])
        d = np.linalg.norm(goal_for_car)

        turn_radius = (d**2)/(2*(goal_for_car[1]))

        steering_angle = math.atan(car_length/turn_radius)

        if steering_angle > 0.4189:
            steering_angle = 0.4189
        elif steering_angle < -0.4189:
            steering_angle = -0.4189
        
        speed = 4.85     

        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steering_angle
        self.ack.drive.speed = speed
        self.ack.header.stamp = rospy.Time.now()
        self.drive_pub.publish(self.ack)

        
def main():
    rospy.init_node('pure_pursuit_node')
    pp = PurePursuit()
    rate = rospy.Rate(7)
    rate.sleep()
    rospy.spin()

if __name__ == '__main__':
    main()
