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

import rospy
import sys
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
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

# goal_x = 259
# goal_y = 33

#Need to update to our maps waypoint file
Waypoint_CSV_File_Path = '/home/zach/catkin_ws/src/lab7/waypoints/centerline_300main.csv'

#Goalpoint Variables
goal_lookahead = 3

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
        #np.set_printoptions(threshold=sys.maxsize)

        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file

        #pf_topic = rospy.get_param('pose_topic')
        #scan_topic = rospy.get_param('scan_topic')

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.tree_pub = rospy.Publisher('/tree_viz_array', MarkerArray, queue_size=1)
        self.map_pub = rospy.Publisher('/map_viz_array', Marker, queue_size=1)
        self.sample_pub = rospy.Publisher('/sample_marker', Marker, queue_size=1)
        self.car_pub = rospy.Publisher('/car_marker', Marker, queue_size=1)
        self.steer_pub = rospy.Publisher('/steer_marker', Marker, queue_size=1)
        self.goal_pub = rospy.Publisher('/goal_marker', Marker, queue_size=1)
        self.path_pub = rospy.Publisher('/path_viz_array', MarkerArray, queue_size=1)
        self.wp_pub = rospy.Publisher('/wp_viz_array', Marker, queue_size=1)
        self.pursuit_pub = rospy.Publisher('/pure_pursuit', Float32MultiArray, queue_size=1)

        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)

        self.iter = 0

        # Importing Waypoints File
        self.Waypoints_Master = np.genfromtxt(Waypoint_CSV_File_Path, delimiter=',', usecols=(0,1), unpack=False)
        #print(self.Waypoints_Master)
        self.goal_lookahead = goal_lookahead

        # class attributes
        # TODO: maybe create your occupancy grid here
        f = open("/home/zach/catkin_ws/src/lab7/maps/300main_clean2.pgm", 'rb')
        unaligned_occ_grid = self.read_pgm(f)
        self.occ_grid = self.align_map(unaligned_occ_grid)

        global width_array
        global height_array

        width_array = np.arange(0, width-1)
        height_array = np.arange(0, height-1)

        #self.unsafe_set = self.get_unsafe_set(occ_grid)
        #self.PublishMapArray(self.occ_grid)
        #self.PublishWaypointArray(self.Waypoints_Master)

        # TODO: create subscribers
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
        car_x = car_x + car_length*math.cos(angle_z)
        car_y = car_y + car_length*math.sin(angle_z)
        car_x, car_y, x_index, y_index = self.localize(car_x, car_y)
        self.PublishCarPoint(car_x, car_y)
        
        car_point = np.asarray([car_x, car_y])
        tree = []
        root_node = Node(x_index, y_index, None, True)
        tree.append(root_node)

        goal_points = self.Waypoints_Master
        #print(goal_points)
        
        goal_magnitudes = np.asarray([LA.norm(goal_point - car_point) for goal_point in goal_points])
        goal_index = self.FindGoalIndex(goal_magnitudes, self.goal_lookahead)

        goal_x, goal_y = self.FindGoalPoint(goal_index, goal_magnitudes, goal_points, self.goal_lookahead)

        goal_found = False

        while goal_found == False:
            #rospy.sleep(0.1)
            sample_point = self.sample(goal_x, goal_y)

            nearest_node = self.nearest(tree, sample_point)
            
            new_node = self.steer_modified(nearest_node, sample_point, tree, x_index, y_index)

            tree_node = Node(new_node[0], new_node[1], nearest_node, False)
            collides = self.check_collision(new_node, nearest_node, tree)
            if collides == False:
                tree.append(tree_node)
                goal_found = self.is_goal(tree_node, goal_x, goal_y)

            
        self.PublishTreeArray(tree)

        unaligned_waypoints = self.find_path(tree, tree_node)

        #print(unaligned_waypoints)

        waypoints = []
        for i in unaligned_waypoints:
            wp_x = self.occ_grid[i[0]][i[1]][1]
            wp_y = self.occ_grid[i[0]][i[1]][2]
            waypoints.append(wp_x)
            waypoints.append(wp_y)
        #print(waypoints)
        pub_points = Float32MultiArray()
        wp_len = len(waypoints)

        dim0 = MultiArrayDimension()
        dim1 = MultiArrayDimension()

        dim0.label = "height"
        dim0.size = wp_len/2
        dim1.label = "width"
        dim1.size = 2

        dimensions = MultiArrayLayout()
        dimensions.dim = [dim0, dim1]

        pub_points.layout = dimensions
        pub_points.data = waypoints

        self.pursuit_pub.publish(pub_points)
        #rospy.sleep(0.1)

        
        #tree_magnitudes = np.asarray([LA.norm(np.asarray([waypoint[0], waypoint[1]]) - car_point) for waypoint in waypoints])
        #tree_index = self.FindGoalIndex(tree_magnitudes, tree_lookahead)
        #tree_x, tree_y = self.FindGoalPoint(tree_index, tree_magnitudes, waypoints, tree_lookahead)

    def func(self, n, i):
        out = 1/float(abs(n-i)+2)
        return out

    def sample(self, goal_x, goal_y):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        denom = 2
        if self.iter % denom == 0 and self.iter != 0:
            self.iter = self.iter + 1
            return np.asarray([goal_x, goal_y])

        is_open = False
        weight_x = lambda n: 1/float(abs(n - goal_x)+2)
        weight_y = lambda n: 1/float(abs(n - goal_y)+2)

        while (is_open == False):

            func_x = np.vectorize(weight_x)
            func_y = np.vectorize(weight_y)

            weighted_x = func_x(width_array)
            weighted_y = func_y(height_array)
            
            sum_x = sum(weighted_x)
            weighted_x = weighted_x/sum_x

            sum_y = sum(weighted_y)
            weighted_y = weighted_y/sum_y
            x = np.random.choice(width_array, p = weighted_x)
            y = np.random.choice(height_array, p = weighted_y)
            #x = random.randint(0, width-1)
            #y = random.randint(0, height-1)

            if self.occ_grid[x][y][0] == 0:
                is_open =True
                #print(angle)



        sx = self.occ_grid[x][y][1]
        sy = self.occ_grid[x][y][2]

        self.PublishSamplePoint(sx, sy)
        sample = np.asarray([x,y])

        self.iter = self.iter + 1

        return sample

    def nearest(self, tree, sample_point):

        magnitudes = np.asarray([LA.norm(np.asarray([tree[i].x, tree[i].y]) - sample_point) for i in range(len(tree))])

        nearest_node = np.argmin(magnitudes)

        return nearest_node

    def steer_modified(self, nearest_node, sample_point, tree, x_index, y_index):
        #rospy.sleep(1)
        node_length = 5

        #print(nearest_node)
        
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

        self.PublishSteerPoint(steer_x, steer_y)

        return new_index

    def check_collision(self, new_node, nearest_node, tree):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        new_x = new_node[0]
        new_y = new_node[1]

        if self.occ_grid[new_x][new_y][0] == 1:
            return True

        near_x = tree[nearest_node].x
        near_y = tree[nearest_node].y
        line_coords = list(bresenham(near_x, near_y, new_x, new_y))

        bubble = (car_width + 0.2) * (1 / map_scalar)

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
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        goal_threshold = 20

        node_point = np.asarray([latest_added_node.x, latest_added_node.y])
        goal_point = np.asarray([goal_x, goal_y])

        dist = LA.norm(node_point - goal_point)

        if dist < goal_threshold:
            return True
        else:
            return False

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
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
                self.PublishPathArray(path_flip)
                return path
            
            parent = node.parent
    
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
        #if x2 == x1:
        #    yi = 
        #else
        #slope = (y2-y1)/(x2-x1)
        #yi = slope*(xi - x1) + y1

        x, y, goal_x, goal_y = self.localize(xi, yi)
        self.PublishGoalPoint(goal_x, goal_y)
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

        angle_z = self.QuatToEuler(qx,qy,qz,qw)

        return car_x, car_y, angle_z

    def PublishTreeArray(self, tree):

        markarray = MarkerArray()
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
        
        id = 0
        for m in markarray.markers:
           m.id = id
           id += 1

        self.tree_pub.publish(markarray)

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

    def PublishMapArray(self, omap):
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

    def align_map(self, occ_grid):
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

    def PublishSamplePoint(self, x, y):
        marker = Marker()
        marker.ns = "sample"
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
        marker.color.g = 0
        marker.color.r = 1.0
        marker.color.b = 0
        marker.color.a = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.sample_pub.publish(marker)

    def PublishCarPoint(self, x, y):
        #print([x, y])
        marker = Marker()
        marker.ns = "car"
        marker.id = 0
        marker.action = marker.MODIFY
        marker.header.frame_id = "/map"
        marker.type = marker.CYLINDER
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.51
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 1
        marker.color.g = 1.0
        marker.color.r = 1.0
        marker.color.b = 0
        marker.color.a = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.car_pub.publish(marker)

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

    def PublishSteerPoint(self, x, y):
        marker = Marker()
        marker.ns = "steer"
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
        marker.color.g = 0
        marker.color.r = 0
        marker.color.b = 1.0
        marker.color.a = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.steer_pub.publish(marker)

    def PublishGoalPoint(self, x_index, y_index):

        node_x = self.occ_grid[x_index][y_index][1]
        node_y = self.occ_grid[x_index][y_index][2]

        marker = Marker()
        marker.ns = "goal"
        marker.id = 0
        marker.action = marker.MODIFY
        marker.header.frame_id = "/map"
        marker.type = marker.CYLINDER
        marker.pose.position.x = node_x
        marker.pose.position.y = node_y
        marker.pose.position.z = 0.51
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 1
        marker.color.g = 0
        marker.color.r = 1
        marker.color.b = 1
        marker.color.a = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.goal_pub.publish(marker)

    def PublishPathArray(self, path):

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

    def PublishWaypointArray(self, waypoints):
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





    

def main():
    rospy.init_node('rrt')
    rrt = RRT()
    rospy.spin()

if __name__ == '__main__':
    main()