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
import tf

# TODO: import as you need
car_width = 0.3
map_scalar = 0.05
map_offset_x = -21.6
map_offset_y = -14.487177

# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = false

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

        # class attributes
        # TODO: maybe create your occupancy grid here
        f = open("/home/zach/catkin_ws/src/lab7/maps/300main_clean2.pgm", 'rb')
        unaligned_occ_grid = self.read_pgm(f)
        self.occ_grid = self.align_map(unaligned_occ_grid)
        #self.unsafe_set = self.get_unsafe_set(occ_grid)
        self.PublishMapArray(self.occ_grid)

        # TODO: create subscribers
        rospy.Subscriber("/odom", Odometry, self.pf_callback)
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """

    def pf_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        car_x, car_y, angle_z = self.ObtainCarState(pose_msg)
        car_x, car_y, x_index, y_index = self.localize_car(car_x, car_y)
        self.PublishCarPoint(car_x, car_y)
        tree = []
        tree.append([0, x_index, y_index])

        goal_found = False
        while goal_found == False:
            #rospy.sleep(1)
            sample_point = self.sample()
            nearest_node = self.nearest(tree, sample_point)
            new_node = self.steer(nearest_node, sample_point, tree, x_index, y_index)
            collides = self.check_collision(new_node, tree)
            if collides == False:
                tree.append((nearest_node, new_node[0], new_node[1]))
                #goal_found = self.is_goal()
            self.PublishTreeArray(tree)
            #print(tree)
            

        return None

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)

        sy = self.occ_grid[x][y][1]
        sx = self.occ_grid[x][y][2]

        self.PublishSamplePoint(sx, sy)
        sample = np.asarray([sx,sy])

        return sample

    def nearest(self, tree, sample_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """

        magnitudes = np.asarray([LA.norm(np.asarray([self.occ_grid[tree[i][1]][tree[i][2]][1],self.occ_grid[tree[i][1]][tree[i][2]][2]]) - sample_point) for i in range(len(tree))])
        #print(magnitudes)

        nearest_node = np.argmin(magnitudes)
        return nearest_node

    def steer(self, nearest_node, sample_point, tree, x_index, y_index):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """

        node_x = tree[nearest_node][1]
        node_y = tree[nearest_node][2]
        sample_x = sample_point[0]
        sample_y = sample_point[1]
        #print([node_x, node_y])

        search_array = [(-1,3),(0,3),(1,3),(-1,-3),(0,-3),(1,-3),(3,1),(3,0),(3,-1),(-3,1),(-3,0),(-3,-1),(2,2),(-2,2),(2,-2),(-2,-2)]
        distances = np.asarray([LA.norm(np.asarray(search_point) - sample_point) for search_point in search_array])
        steer_index = np.argmin(distances)
        
        new_x_index = node_x + search_array[steer_index][0]
        new_y_index = node_y + search_array[steer_index][1]

        new_index = (new_x_index, new_y_index)

        steer_x = self.occ_grid[new_x_index][new_y_index][1]
        steer_y = self.occ_grid[new_x_index][new_y_index][2]

        self.PublishSteerPoint(steer_x, steer_y)

        return new_index

    def check_collision(self, new_node, tree):
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
        x_index = new_node[0]
        y_index = new_node[1]
        collides = self.occ_grid[x_index][y_index][0]
        if collides == 0:
            return True
        else:
            return False

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
        path = []
        return path



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
                    raster[i][j] = 0
                else:
                    raster[i][j] = 1
                    

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

    def get_unsafe_set(self, occ_grid):
        #depricated
        unsafe_set = []
        for i in range(height):
            for j in range(width):
                if occ_grid[i][j] == 1:
                    unsafe_set.append((i*0.05, j*0.05))
        #print(unsafe_set)

        return unsafe_set

    def PublishTreeArray(self, tree):

        markarray = MarkerArray()
        for i in range(len(tree)):
            line_point = self.occ_grid[tree[i][1]][tree[i][2]]
            line_x = line_point[1]
            line_y = line_point[2]
            points = self.ConstructPointsArray(line_x, line_y)

            marker = Marker()
            marker.ns = "waypoints"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.LINE_STRIP
            marker.points = points
            marker.scale.x = 0.01
            marker.color.g = 1.0
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

        self.tree_pub.publish(markarray)

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
                    p.z = 0.1
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

    def localize_car(self, x, y):
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





    

def main():
    rospy.init_node('rrt')
    rrt = RRT()
    rospy.spin()

if __name__ == '__main__':
    main()