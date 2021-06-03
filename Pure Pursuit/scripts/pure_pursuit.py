#!/usr/bin/env python
import rospy
import sys
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import numpy as np
import math

# TODO: import ROS msg types and libraries
LOOKAHEAD = 4.6
Waypoint_CSV_File_Path = '/home/autocar/catkin_ws/src/scripts/pure_pur/waypoints/raceline.csv'
Odom_Topic = '/pf/pose/odom'
Car_Length = 0.32

class PurePursuit(object):
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        global Waypoint_CSV_File_Path
        global LOOKAHEAD
        global Car_Length

        self.L = LOOKAHEAD
        self.car_length = Car_Length
        self.Waypoints_Master = np.loadtxt(Waypoint_CSV_File_Path, delimiter=',', dtype=None, usecols=(0,1), unpack=False)

        np.set_printoptions(threshold=sys.maxsize)

        self.pose_sub = rospy.Subscriber(Odom_Topic, Odometry, self.pose_callback, queue_size=1) #TODO
        self.goal_pub = rospy.Publisher('/goalpoint', Marker, queue_size=10)
        #self.wp_pub = rospy.Publisher('/waypoint_viz_array', MarkerArray, queue_size=1) #TODO
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
        
    def ConstructPointsArray(self, x_coord, y_coord):
        p1 = Point()
        p1.x = x_coord
        p1.y = y_coord
        p1.z = 0
        
        p2 = Point()
        p2.x = x_coord
        p2.y = y_coord
        p2.z = 1

        pointlist = [p1, p2]

        return pointlist

    def ConstructMarkerArray(self, waypoints):
        global iteration

        markarray = MarkerArray()
        for i in [0,len(waypoints)-1]:
            line_x = waypoints[i][0]
            line_y = waypoints[i][1]
            points = self.ConstructPointsArray(line_x, line_y)

            marker = Marker()
            marker.ns = "waypoints"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.LINE_LIST
            marker.points = points
            marker.scale.x = 0.1
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration(1)
            markarray.markers.append(marker)
        
        id = 0
        for m in markarray.markers:
           m.id = id
           id += 1

        return markarray
    
    def PublishMarkerArray(self, waypoints):

        markarray = self.ConstructMarkerArray(waypoints)
        #self.wp_pub.publish(markarray)

    def QuatToEuler(self,x,y,z,w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        rad = (np.arctan2(siny_cosp, cosy_cosp))
        return rad

    def ObtainCarState(self, pose_msg):
        car_x = pose_msg.pose.pose.position.x
        car_y = pose_msg.pose.pose.position.y

        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w

        car_point = np.asarray([car_x, car_y])

        angle_z = self.QuatToEuler(qx,qy,qz,qw)

        return car_point, angle_z

    def PublishGoalPoint(self, goal_point):
        p1 = Point()
        p1.x = goal_point[0]
        p1.y = goal_point[1]
        p1.z = 0
        
        p2 = Point()
        p2.x = goal_point[0]
        p2.y = goal_point[1]
        p2.z = 1

        points = [p1, p2]

        marker = Marker()
        marker.id = 0
        marker.ns = "goalpoint"
        marker.action = marker.MODIFY
        marker.header.frame_id = "/map"
        marker.type = marker.LINE_LIST
        marker.points = points
        marker.scale.x = 0.1
        marker.color.g = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.goal_pub.publish(marker)

    def PublishTransformedGoalPoint(self, goal_point):
        p1 = Point()
        p1.x = goal_point[0]
        p1.y = goal_point[1]
        p1.z = 0
        
        p2 = Point()
        p2.x = goal_point[0]
        p2.y = goal_point[1]
        p2.z = 1

        points = [p1, p2]

        marker = Marker()
        marker.id = 0
        marker.ns = "translatedgoalpoint"
        marker.action = marker.MODIFY
        marker.header.frame_id = "/base_link"
        marker.type = marker.LINE_LIST
        marker.points = points
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.goal_pub.publish(marker)

    def FindGoalIndex(self, distances, L):
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

    def FindGoalPoint(self, goal_index, magnitudes, waypoints, L):
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
        slope = (y2-y1)/(x2-x1)
        yi = slope*(xi - x1) + y1

        goal_point = np.asarray([xi,yi])
        return goal_point

    def pose_callback(self, pose_msg): 
        L = self.L
        car_length = self.car_length
        waypoints = self.Waypoints_Master

        #TODO: obtain current car state
        car_point, angle_z = self.ObtainCarState(pose_msg)

        # TODO: find the current waypoint to track using methods mentioned in lecture
        magnitudes = np.asarray([np.linalg.norm(waypoints[i] - car_point) for i in range(len(waypoints))])

        goal_index = self.FindGoalIndex(magnitudes, L)

        goal_point = self.FindGoalPoint(goal_index, magnitudes, waypoints, L)

        
        #self.PublishGoalPoint(goal_point)

        # TODO: transform goal point to vehicle frame of reference
        x = (goal_point[0] - car_point[0])*math.cos(angle_z) + (goal_point[1] - car_point[1])*math.sin(angle_z)
        y = -(goal_point[0] - car_point[0])*math.sin(angle_z) + (goal_point[1] - car_point[1])*math.cos(angle_z)

        goal_for_car = np.asarray([x, y])
        d = np.linalg.norm(goal_for_car)

        #self.PublishTransformedGoalPoint(goal_for_car)

        # TODO: calculate curvature/steering angle

        turn_radius = (d**2)/(2*(goal_for_car[1]))

        steering_angle = math.atan(car_length/turn_radius)

        # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians

        if steering_angle > 0.4189:
            steering_angle = 0.4189
        elif steering_angle < -0.4189:
            steering_angle = -0.4189

	    if (abs(steering_angle) > 0.065):
	        speed = 2 
	    elif (abs(steering_angle) > 0.04) and (abs(steering_angle) <= 0.065):
	        speed = 3
	    else:
	        speed = 5

        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steering_angle
        self.ack.drive.speed = speed
        self.ack.header.stamp = rospy.Time.now()
        
        self.drive_pub.publish(self.ack)

        
def main():
    rospy.init_node('pure_pursuit_node')
    pp = PurePursuit()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main()
