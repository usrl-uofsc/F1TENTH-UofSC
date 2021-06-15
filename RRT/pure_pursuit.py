#!/usr/bin/env python
import rospy
import sys
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import numpy as np
import math

# TODO: import ROS msg types and libraries
LOOKAHEAD = 1.2
Waypoint_CSV_File_Path = '/home/zach/catkin_ws/src/lab6/waypoints/levine-waypoints.csv'
Odom_Topic = '/odom'
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

        np.set_printoptions(threshold=sys.maxsize)

        rospy.Subscriber("/pure_pursuit", Float32MultiArray, self.pose_callback, queue_size=1) #TODO
        
        self.goal_pub = rospy.Publisher('/goalpoint', Marker, queue_size=10)
        #self.wp_pub = rospy.Publisher('/waypoint_viz_array', MarkerArray, queue_size=1) #TODO
        self.drive_pub = rospy.Publisher('drive', AckermannDriveStamped, queue_size=1)
        
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

    def ObtainCarState(self, pose_msg, car_length):
        car_x = pose_msg.pose.pose.position.x
        car_y = pose_msg.pose.pose.position.y

        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w

        angle_z = self.QuatToEuler(qx,qy,qz,qw)

        car_x = car_x + car_length*math.cos(angle_z)
        car_y = car_y + car_length*math.sin(angle_z)

        car_point = np.asarray([car_x, car_y])

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
        yi = np.interp(mi, [m1,m2], [y1, y2])

        goal_point = np.asarray([xi,yi])
        return goal_point

    # def rrt_callback(self, wp_msg):
        height = wp_msg.layout.dim[0].size
        width = wp_msg.layout.dim[1].size
        data = np.asarray(wp_msg.data)
        self.Waypoints_Master = np.reshape(data, (height, width))
        rospy.Subscriber(Odom_Topic, Odometry, self.pose_callback, queue_size=1) #TODO 

    def pose_callback(self, wp_msg): 
        height = wp_msg.layout.dim[0].size
        width = wp_msg.layout.dim[1].size
        data = np.asarray(wp_msg.data)
        self.Waypoints_Master = np.reshape(data, (height, width))

        L = self.L
        #print("check")
        car_length = self.car_length
        waypoints = self.Waypoints_Master[0:, 3:]
        print(waypoints)
        #TODO: obtain current car state
        car_point = self.Waypoints_Master[0, 0:2]
        angle_z = self.Waypoints_Master[0, 2]
        # car_point, angle_z = self.ObtainCarState(pose_msg, car_length)
        print(car_point)
        print(angle_z)
        # TODO: find the current waypoint to track using methods mentioned in lecture
        magnitudes = np.asarray([np.linalg.norm(waypoint - car_point) for waypoint in waypoints])

        goal_index = self.FindGoalIndex(magnitudes, L)
        #goal_point = waypoints[goal_index]
        #goal_magnitude = magnitudes[goal_index]

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
        #print(steering_angle)

        # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians

        if steering_angle > 0.4189:
            steering_angle = 0.4189
        elif steering_angle < -0.4189:
            steering_angle = -0.4189
        
        # print(steering_angle)
        '''
        if (abs(steering_angle) > 0.065):
	        speed = 4.5 
        elif (abs(steering_angle) > 0.04) and (abs(steering_angle) <= 0.065):
	        speed = 5
        else:
	        speed = 7
        '''

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
    rospy.spin()

if __name__ == '__main__':
    main()