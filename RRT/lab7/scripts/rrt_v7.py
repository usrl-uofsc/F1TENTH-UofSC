#!/usr/bin/env python
"""
ESE 680
RRT assignment
Author: Hongrui Zheng

This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math
import random
import time

from numpy import ma

import rospy
import sys
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose, Twist
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import tf
from bresenham import bresenham

# TODO: import as you need
car_width = 0.3
car_length = 0.32
map_scalar = 0.05
map_offset_x = -21.6
map_offset_y = -14.487177

LOOKAHEAD = 0.5
Odom_Topic = '/odom'
Car_Length = 0.32

# goal_x = 259
# goal_y = 33

#Need to update to our maps waypoint file
Waypoint_CSV_File_Path = '/home/zach/catkin_ws/src/lab7/waypoints/centerline_300main.csv'

#Goalpoint Variables
goal_lookahead = 4

iter_ = 0
total = 0

# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self, x, y, parent, is_root):
        self.x = x
        self.y = y
        self.parent = parent
        #self.cost = None # only used in RRT*
        self.is_root = is_root

# class def for RRT
class RRT(object):
    def __init__(self):

        """Importing Waypoints File"""
        self.Waypoints_Master = np.genfromtxt(Waypoint_CSV_File_Path, delimiter=',', usecols=(0,1), unpack=False)
        #print(self.Waypoints_Master)
        self.goal_lookahead = goal_lookahead

        global total
        global iter_

        self.total = total
        self.iter = iter_

        """class attributes"""
        f = open("/home/zach/catkin_ws/src/lab7/maps/300main_clean2.pgm", 'rb')
        unaligned_occ_grid = self.read_pgm(f)
        self.occ_grid = self.AlignMap(unaligned_occ_grid)
        print(self.occ_grid)
        self.Visualize = Visualize(self.occ_grid)
        self.PurePursuit = PurePursuit(self.occ_grid)

        global width_array
        global height_array

        width_array = np.arange(0, width-1)
        height_array = np.arange(0, height-1)

        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)

        """Subscribers"""
        rospy.Subscriber("/odom", Odometry, self.pf_callback, queue_size=1)
        #self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, buff_size=1)
        
    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """

    def pf_callback(self, pose_msg):
        car_x, car_y, angle_z = self.ObtainCarState(pose_msg)
        car_x, car_y, car_x_index, car_y_index = self.localize(car_x, car_y)
        self.Visualize.Point.Car(car_x, car_y)
        car_point = np.asarray([car_x, car_y])

        tree = []
        root_node = Node(car_x_index, car_y_index, None, True)
        tree.append(root_node)

        goal_points = self.Waypoints_Master
        goal_magnitudes = np.asarray([LA.norm(goal_point - car_point) for goal_point in goal_points])
        goal_index = self.find_goal_index(goal_magnitudes, self.goal_lookahead)
        goal_x, goal_y = self.find_goal_point(goal_index, goal_magnitudes, goal_points, self.goal_lookahead)

        goal_found = False

        weighted_x, weighted_y = self.ConstructWeightArrays(goal_x, goal_y)
        
        while goal_found == False:
            sample_point = self.sample(goal_x, goal_y, weighted_x, weighted_y)
            nearest_node = self.nearest(tree, sample_point)

            self.node_length = 20
            new_node = self.steer(nearest_node, sample_point, tree, car_x_index, car_y_index, self.node_length)
            collides = self.check_collision(new_node, nearest_node, tree)

            if collides == False:
                tree_node = Node(new_node[0], new_node[1], nearest_node, False)
                tree.append(tree_node)
                goal_found = self.is_goal(tree_node, goal_x, goal_y)

        self.Visualize.Line.TreeArray(tree)

        unaligned_waypoints = self.find_path(tree, tree_node)
        waypoints = []
        for i in unaligned_waypoints:
            wp_x = self.occ_grid[i[0]][i[1]][1]
            wp_y = self.occ_grid[i[0]][i[1]][2]
            wp = (wp_x, wp_y)
            waypoints.append(wp)
        
        self.PurePursuit.Drive(waypoints, car_point, angle_z)

    def sample(self, goal_x, goal_y, weighted_x, weighted_y):
        denom = 2
        if self.iter % denom == 0 and self.iter != 0:
            self.iter = self.iter + 1
            return np.asarray([goal_x, goal_y])

        is_open = False

        while (is_open == False):
            x = np.random.choice(width_array, p = weighted_x)
            y = np.random.choice(height_array, p = weighted_y)

            if self.occ_grid[x][y][0] == 0:
                is_open =True
                #print(angle)

        sx = self.occ_grid[x][y][1]
        sy = self.occ_grid[x][y][2]

        self.Visualize.Point.Sample(sx, sy)
        sample = np.asarray([x,y])

        self.iter = self.iter + 1

        return sample

    def nearest(self, tree, sample_point):

        magnitudes = np.asarray([LA.norm(np.asarray([tree[i].x, tree[i].y]) - sample_point) for i in range(len(tree))])

        nearest_node = np.argmin(magnitudes)

        return nearest_node

    def steer(self, nearest_node, sample_point, tree, x_index, y_index, node_length):
        
        node_x = tree[nearest_node].x
        node_y = tree[nearest_node].y
        sample_x = sample_point[0]
        sample_y = sample_point[1]

        node_point = np.asarray([node_x, node_y])

        dist = LA.norm(node_point - sample_point)

        new_x_index = int(round(np.interp(node_length, [0,dist], [node_x, sample_x])))
        new_y_index = int(round(np.interp(node_length, [0,dist], [node_y, sample_y])))

        new_index = (new_x_index, new_y_index)

        steer_x = self.occ_grid[new_x_index][new_y_index][1]
        steer_y = self.occ_grid[new_x_index][new_y_index][2]

        self.Visualize.Point.Steer(steer_x, steer_y)

        return new_index

    def check_collision(self, new_node, nearest_node, tree):
        new_x = new_node[0]
        new_y = new_node[1]

        if self.occ_grid[new_x][new_y][0] == 1:
            return True

        near_x = tree[nearest_node].x
        near_y = tree[nearest_node].y
        line_coords = list(bresenham(near_x, near_y, new_x, new_y))

        bubble = (car_width) * (1/map_scalar)

        bound = math.sqrt(2)/2

        if (near_x == new_x):
            angle = math.pi/2
        else:
            m = (new_y - near_y)/(new_x - near_x)
            angle = math.atan(m)

        sum_ = 0

        buffer = (bubble/2)*math.cos(angle) + bound
        for i in line_coords:
            y_coord = i[1]
            lower = int(math.floor(y_coord - buffer))
            upper = int(math.ceil(y_coord + buffer))
            for j in range(lower, upper):
                x = i[0]
                collides = self.occ_grid[x][j][0]
                sum_ = sum_ + collides


            x_coord = i[0]
            lower = int(math.floor(x_coord - buffer))
            upper = int(math.ceil(x_coord + buffer))
            for j in range(lower, upper):
                y = i[1]
                collides = self.occ_grid[j][y][0]
                sum_ = sum_ + collides

        if sum_ == 0:
            #print("occ")
            return False

        return True

    def is_goal(self, latest_added_node, goal_x, goal_y):
        goal_threshold = 17

        node_point = np.asarray([latest_added_node.x, latest_added_node.y])
        goal_point = np.asarray([goal_x, goal_y])

        dist = LA.norm(node_point - goal_point)

        if dist < goal_threshold:
            return True
        else:
            return False

    def find_path(self, tree, latest_added_node):
        path = np.empty((0,2), dtype=tuple)
        end = (int(latest_added_node.x), int(latest_added_node.y))
        path = np.append(path, end)
        parent = latest_added_node.parent
        root = False
        while root == False:
            node = tree[parent]
            path = np.append(path, (int(node.x), int(node.y)))

            if node.is_root == True:
                path = np.reshape(path, (len(path)/2, 2))
                path_flip = np.flip(path,0)
                self.Visualize.Line.PathArray(path_flip)
                return path
            
            parent = node.parent
    
    def find_goal_index(self, distances, L):
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

    def find_goal_point(self, goal_index, magnitudes, waypoints, L):
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

        x, y, goal_x, goal_y = self.localize(xi, yi)
        self.Visualize.Point.Goal(goal_x, goal_y)
        return goal_x, goal_y

    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

    #Additional Functions to Aid in Data Management
    def read_pgm(self, pgmf):
        """Return a raster of integers from a PGM as a list of lists."""
        assert pgmf.readline() == 'P5\n'
        global width
        global height
        width, height = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())
        assert depth <= 255

        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(pgmf.read(1)))
            raster.append(row)

        raster = np.asarray(raster, dtype=object)
        for i in range(len(raster)):
            for j in range(len(raster[i])):
                """ '1' means 'ocuppied', '0' means 'open' """
                if raster[i][j] < 250:
                    raster[i][j] = 1
                else:
                    raster[i][j] = 0
                    
        #self.print_raster_to_terminal(raster)

        return raster

    def print_raster_to_terminal(self, raster):
        s = 220

        for i in range(len(raster)):
            list = []
            for j in range(0+s,306+s):
                list.append(raster[i][j])
            print(list)

    def QuatToEuler(self,x,y,z,w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        rad = (math.atan2(siny_cosp, cosy_cosp))
        return rad

    def ObtainCarState(self, pose_msg):
        car_x = pose_msg.pose.pose.position.x
        car_y = pose_msg.pose.pose.position.y

        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w

        vx = pose_msg.twist.twist.linear.x
        vy = pose_msg.twist.twist.linear.y

        vel = math.sqrt((vx**2)+(vy**2))

        self.total += vel
        self.iter += 1

        avg = (self.total/self.iter)

        #print(avg)

        angle_z = self.QuatToEuler(qx,qy,qz,qw)

        """Moves car reference point from rear to front"""
        car_x = car_x + car_length*math.cos(angle_z)
        car_y = car_y + car_length*math.sin(angle_z)

        return car_x, car_y, angle_z

    def localize(self, x, y):
        x_sig_figs = len(str(map_offset_x))-2
        y_sig_figs = len(str(map_offset_y))-2 
        x_index = int(round((x - map_offset_x)/map_scalar))
        y_index = int(round((y - map_offset_y)/map_scalar))
        #print(x_index, y_index)
        x = (round((x - map_offset_x)/map_scalar) * map_scalar) + map_offset_x
        y = (round((y - map_offset_y)/map_scalar) * map_scalar) + map_offset_y
        x = round(x, x_sig_figs)
        y = round(y, y_sig_figs)
        return x, y, x_index, y_index

    def AlignMap(self, occ_grid):
        occ_grid = np.transpose(occ_grid)
        occ_grid = np.flip(occ_grid, 1)
        x_sig_figs = len(str(map_offset_x))-2
        y_sig_figs = len(str(map_offset_y))-2
        for i in range(width):
            for j in range(height):
                depth = occ_grid[i][j]
                x = i * map_scalar + map_offset_x
                x = round(x, x_sig_figs)
                y = j * map_scalar + map_offset_y
                y = round(y, y_sig_figs)
                data = [depth, x ,y]
                #if data[1] == 0:
                #    print(data)
                occ_grid[i][j] = data

        return occ_grid

    def PublishtoPurePursuit(self, waypoints):
        pub_points = Float32MultiArray()
        wp_len = len(waypoints)

        dim0 = MultiArrayDimension()
        dim1 = MultiArrayDimension()

        dim0.label = "height"
        dim0.size = wp_len/3
        dim1.label = "width"
        dim1.size = 3

        dimensions = MultiArrayLayout()
        dimensions.dim = [dim0, dim1]

        pub_points.layout = dimensions
        pub_points.data = waypoints

        self.pursuit_pub.publish(pub_points)

    def func(self, n, i):
        out = 1/float(abs(n-i)+2)
        return out

    def ConstructWeightArrays(self, goal_x, goal_y):
        weight_x = lambda n: 1/float(abs(n - goal_x)+2)
        weight_y = lambda n: 1/float(abs(n - goal_y)+2)

        func_x = np.vectorize(weight_x)
        func_y = np.vectorize(weight_y)

        weighted_x = func_x(width_array)
        weighted_y = func_y(height_array)

        sum_x = sum(weighted_x)
        weighted_x = weighted_x/sum_x

        sum_y = sum(weighted_y)
        weighted_y = weighted_y/sum_y

        return weighted_x, weighted_y

class PurePursuit(object):
    def __init__(self, occ_grid):
        global LOOKAHEAD
        global Car_Length

        self.L = LOOKAHEAD
        self.car_length = Car_Length

        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)

        #self.iter = 0
        self.Visualize = Visualize(occ_grid)

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

        index = int((len(waypoints)/2))

        xi = waypoints[index][0]
        yi = waypoints[index][1]

        goal_point = np.asarray([xi,yi])
        return goal_point

    def Drive(self, waypoints, car_point, angle_z):
        #print(self.iter)
        #self.iter += 1
        
        L = self.L

        car_length = self.car_length

        magnitudes = np.asarray([np.linalg.norm(waypoint - car_point) - L for waypoint in waypoints])

        goal_index = self.FindNavIndex(magnitudes, L)

        goal_point = self.FindNavPoint(goal_index, magnitudes, waypoints, L)
        
        x = (goal_point[0] - car_point[0])*math.cos(angle_z) + (goal_point[1] - car_point[1])*math.sin(angle_z)
        y = -(goal_point[0] - car_point[0])*math.sin(angle_z) + (goal_point[1] - car_point[1])*math.cos(angle_z)

        self.Visualize.Point.Nav(goal_point[0], goal_point[1])

        goal_for_car = np.asarray([x, y])
        d = np.linalg.norm(goal_for_car)

        turn_radius = (d**2)/(2*(goal_for_car[1]))

        steering_angle = math.atan(car_length/turn_radius)

        s = 0.4814

        if steering_angle > s:
            steering_angle = s
        elif steering_angle < -s:
            steering_angle = -s
            
        #if (abs(steering_angle) > 0.25):
        #    speed = 2
        #elif (abs(steering_angle) > 0.15) and (abs(steering_angle) <= 0.25):
        #    speed = 3
        #else:
        #    speed = 5

        m = 7
        n = 3

        speed = (m-n)*(s - abs(steering_angle)) + n

        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steering_angle
        self.ack.drive.speed = speed
        self.ack.header.stamp = rospy.Time.now()
        self.drive_pub.publish(self.ack)

class Visualize(object):
    def __init__(self, occ_grid):
        self.Point = self.Point(occ_grid)
        self.Line = self.Line(occ_grid)
        self.Array = self.Array()

    class Array(object):
        def __init__(self):
            self.occ_grid_pub = rospy.Publisher('/occupancy_grid_array', Marker, queue_size=1)
            self.wp_pub = rospy.Publisher('/waypoint_array', Marker, queue_size=1)

        def OccupancyGrid(self, omap):
            points = []
            for i in range(width):
                for j in range(height):
                    if omap[i][j][0] == 1:
                        line_x = omap[i][j][1]
                        line_y = omap[i][j][2]
                        p = Point()
                        p.x = line_x
                        p.y = line_y
                        p.z = 0.01
                        points.append(p)

            marker = Marker()
            marker.ns = "map"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 1.0
            marker.color.r = 1.0
            marker.color.b = 1.0
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            #marker.lifetime = rospy.Duration(1)

            self.map_pub.publish(marker)

        def WaypointArray(self, waypoints):
            points = []
            for i in waypoints:
                p = Point()
                p.x = i[0]
                p.y = i[1]
                p.z = 0.01
                points.append(p)            

            marker = Marker()
            marker.ns = "waypoints"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 0
            marker.color.r = 1.0
            marker.color.b = 1.0
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            #marker.lifetime = rospy.Duration(1)

            self.wp_pub.publish(marker)

    class Point(object):
        def __init__(self, occ_grid):
            self.sample_pub = rospy.Publisher('/sample_point_marker', Marker, queue_size=1)
            self.car_pub = rospy.Publisher('/car_point_marker', Marker, queue_size=1)
            self.steer_pub = rospy.Publisher('/steer_point_marker', Marker, queue_size=1)
            self.goal_pub = rospy.Publisher('/goal_point_marker', Marker, queue_size=1)
            self.nav_pub = rospy.Publisher('/navigation_point_marker', Marker, queue_size=1)
            self.occ_grid = occ_grid

        def CreatePointMarker(self, namespace, x, y, red, green, blue):
            marker = Marker()
            marker.ns = namespace
            marker.id = 0
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.CYLINDER
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.51
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 1
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 1
            marker.pose.orientation.w = 1.0

            return marker

        def Sample(self, x, y):
            namespace = "sample"
            red = 1
            green = 0
            blue = 0
            
            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.sample_pub.publish(marker)

        def Nav(self, x, y):
            namespace = "nav"
            red = 0
            green = 1
            blue = 1
            
            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.nav_pub.publish(marker)

        def Car(self, x, y):
            namespace = "car"
            green = 1
            red = 1
            blue = 0

            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.car_pub.publish(marker)

        def Steer(self, x, y):
            namespace = "steer"
            red = 0
            green = 0
            blue = 1

            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.steer_pub.publish(marker)

        def Goal(self, x_index, y_index):
            x = self.occ_grid[x_index][y_index][1]
            y = self.occ_grid[x_index][y_index][2]

            namespace = "goal"
            red = 1
            green = 0
            blue = 1

            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.goal_pub.publish(marker)

    class Line(object):
        def __init__(self, occ_grid):
            self.occ_grid = occ_grid
            self.tree_pub = rospy.Publisher('/tree_viz_array', MarkerArray, queue_size=1)
            self.path_pub = rospy.Publisher('/path_viz_array', MarkerArray, queue_size=1)

        def ConstructPointsArray(self, nx, ny, px, py, z):
            p1 = Point()
            p1.x = nx
            p1.y = ny
            p1.z = z

            p2 = Point()
            p2.x = px
            p2.y = py
            p2.z = z

            pointlist = [p1, p2]

            return pointlist

        def TreeArray(self, tree):

            markarray = MarkerArray()
            id_ = 0
            for i in range(1,len(tree)):
                x_index = tree[i].x
                y_index = tree[i].y

                parent_index = tree[i].parent
                parent_x = tree[parent_index].x
                parent_y = tree[parent_index].y

                node_x = self.occ_grid[x_index][y_index][1]
                node_y = self.occ_grid[x_index][y_index][2]

                pnode_x = self.occ_grid[parent_x][parent_y][1]
                pnode_y = self.occ_grid[parent_x][parent_y][2]

                z = 0.025

                points = self.ConstructPointsArray(node_x, node_y, pnode_x, pnode_y, z)

                marker = Marker()
                marker.id = id_
                marker.ns = "tree"
                marker.action = marker.MODIFY
                marker.header.frame_id = "/map"
                marker.type = marker.LINE_STRIP
                marker.points = points
                marker.scale.x = 0.05
                marker.color.g = 1.0
                marker.color.a = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.lifetime = rospy.Duration(0.1)
                markarray.markers.append(marker)
                id_ += 1

            self.tree_pub.publish(markarray)

        def PathArray(self, path):

            markarray = MarkerArray()
            for i in range(1,len(path)):
                #print(path)
                x_index = path[i][0]
                y_index = path[i][1]

                parent_x = path[i-1][0]
                parent_y = path[i-1][1]

                node_x = self.occ_grid[x_index][y_index][1]
                node_y = self.occ_grid[x_index][y_index][2]

                pnode_x = self.occ_grid[parent_x][parent_y][1]
                pnode_y = self.occ_grid[parent_x][parent_y][2]

                z = 0.05

                points = self.ConstructPointsArray(node_x, node_y, pnode_x, pnode_y, z)

                marker = Marker()
                marker.ns = "path"
                marker.action = marker.MODIFY
                marker.header.frame_id = "/map"
                marker.type = marker.LINE_STRIP
                marker.points = points
                marker.scale.x = 0.05
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.lifetime = rospy.Duration(2)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                markarray.markers.append(marker)

            id = 0
            for m in markarray.markers:
               m.id = id
               id += 1

            self.path_pub.publish(markarray)

def main():
    rospy.init_node('rrt')
    rrt = RRT()
    rate = rospy.Rate(12.5)
    rate.sleep()
    rospy.spin()

if __name__ == '__main__':
    main()