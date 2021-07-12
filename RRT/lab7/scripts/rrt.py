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
from geometry_msgs.msg import Point, Point32
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import tf
from itertools import product
from skimage.draw import line, circle, ellipse

# TODO: import as you need
car_width = rospy.get_param("/car_width")
car_length = rospy.get_param("/car_length")
map_scalar = rospy.get_param("/map_scalar")
map_offset_x = rospy.get_param("/map_offset_x")
map_offset_y = rospy.get_param("/map_offset_y")

LOOKAHEAD = rospy.get_param("/LOOKAHEAD")
Odom_Topic = rospy.get_param("/pose_topic")

#Need to update to our maps waypoint file
Waypoint_CSV_File_Path = rospy.get_param("/Waypoint_CSV_File_Path")

#Goalpoint Variables
goal_lookahead = rospy.get_param("/goal_lookahead")
Occupancy_Grid_File_Path = rospy.get_param("/Occupancy_Grid_File_Path")
drive_topic = rospy.get_param("/drive_topic")

iter_ = 0
total = 0

# class def for RRT
class RRT(object):
    def __init__(self):

        """Importing Waypoints File"""
        self.Waypoints_Master = np.genfromtxt(Waypoint_CSV_File_Path, delimiter=',', usecols=(0,1), unpack=False)
        self.goal_lookahead = goal_lookahead
        self.bound = rospy.get_param("/bubble_extra_bound")
        self.node_length = rospy.get_param("/node_length")
        neighborhood_radius = rospy.get_param("/neighborhood_radius")
        self.goal_threshold = rospy.get_param("/goal_threshold")
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1)

        global total
        global iter_

        np.set_printoptions(threshold=sys.maxsize)

        self.total = total
        self.iter = iter_
        self.made_angle_array = False

        self.buffer = int(math.ceil((car_width/2) * (1/map_scalar) + self.bound))
        self.scan_buffer = np.transpose(circle(0, 0, self.buffer))

        """vectorizers"""
        self.localize_scan_vectorized = np.vectorize(self.localize_scan)
        self.localize_car_vectorized = np.vectorize(self.localize_car)
        self.transform_lasers_vectorized = np.vectorize(self.transform_lasers)
        self.scan_line_clear_vectorized = np.vectorize(self.ScanLineClear)
        self.scan_buffer_brens_vectorized = np.vectorize(self.scan_buffer_brens, otypes=[np.ndarray])
        self.near_check_vectorized = np.vectorize(self.near_check)

        """class attributes"""
        f = open(Occupancy_Grid_File_Path, 'rb')
        occ_grid = self.read_pgm(f)
        self.transform_grid = self.AlignMap(occ_grid)
        safe_set = np.where(occ_grid == 0)
        self.safe_set = np.column_stack((safe_set[0], safe_set[1]))
        self.occ_grid, self.global_safe_set = self.AddBuffertoMap(occ_grid, self.safe_set)

        self.neighborhood_stamp = np.transpose(circle(0, 0, neighborhood_radius))
        
        self.Visualize = Visualize(self.transform_grid)
        self.PurePursuit = PurePursuit(self.transform_grid)

        rospy.sleep(0.5)

        self.Visualize.Array.OccupancyGrid(self.global_safe_set)
        self.Visualize.Array.WaypointArray(self.Waypoints_Master)

        self.global_occ_grid = np.copy(self.occ_grid)
        self.r = 2*self.goal_lookahead + (self.buffer*map_scalar)
        self.ellipse_width = (self.goal_lookahead/map_scalar)
        self.ellipse_height = (self.goal_lookahead/map_scalar)

        self.empty = np.multiply(self.occ_grid, 0)

        """Subscribers"""
        rospy.Subscriber(Odom_Topic, Odometry, self.pf_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        
    def scan_callback(self, scan_msg):
        if self.made_angle_array == False:
            self.angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)[179:899]
            self.made_angle_array = True

        distances = np.asarray(scan_msg.ranges)[179:899]
        self.distances = np.clip(distances, 0, self.r)

        self.laser_bools = distances < self.r

    def pf_callback(self, pose_msg):
        car_point, angle_z = self.ObtainCarState(pose_msg)
        car_index = self.localize_car_vectorized(car_point[0], car_point[1])
        car_index = np.asarray(car_index)
        self.Visualize.Point.Car(car_point)

        self.UpdateOccupancyGrid(car_point, car_index, angle_z)

        if self.occ_grid[car_index[0]][car_index[1]] == 1:
            t = np.add(self.scan_buffer, car_index)
            self.occ_grid[t[:,0], t[:,1]] = 0

        goal_magnitudes = LA.norm(self.Waypoints_Master - car_point, axis = 1) #0.0001
        goal_index = self.find_goal_index(goal_magnitudes, self.goal_lookahead) #0.0003
        goal_x_index, goal_y_index, goal_x, goal_y = self.find_goal_point(goal_index, goal_magnitudes, self.Waypoints_Master, self.goal_lookahead)#0.0002
        goal = np.asarray([goal_x_index, goal_y_index])

        mx = int((goal_x_index + car_index[0])/2)
        my = int((goal_y_index + car_index[1])/2)

        sample_space = np.transpose(ellipse(mx, my, self.ellipse_height, self.ellipse_width, (611, 408)))
        sample_space = sample_space[self.occ_grid[sample_space[:,0], sample_space[:,1]] == 0]
        #print(sample_area)

        self.Visualize.Array.Sample(sample_space[:,0], sample_space[:,1])

        E = np.empty((0, 4))
        car_x_index = car_index[0]
        car_y_index = car_index[1]
        root_node = np.asarray([car_x_index, car_y_index, None, True])
        E = np.vstack((E, root_node))

        V = np.empty((0,2), dtype='int')
        V = np.vstack((V, car_index))

        costs = np.empty((0,1))
        costs = np.append(costs, 0)

        goal_found = False
        while not goal_found:
            x_rand = self.sample(sample_space)
            x_near, near_index = self.nearest(V, x_rand)
            x_new = self.steer(x_rand, x_near, self.node_length)
            collides = self.check_collision(x_new, x_near)
            if not collides:
                neighbors = self.near(V, x_new)
                neighbor_points = V[neighbors]
                neighbor_collisions = self.near_check_vectorized(neighbor_points[:,0], neighbor_points[:,1], x_new[0], x_new[1])
                neighbors = neighbors[neighbor_collisions]

                parent, cost = self.cost(neighbors, x_new, E, V, costs)

                V = np.vstack((V, x_new))
                node = np.asarray([x_new[0], x_new[1], parent, False])
                E = np.vstack((E, node))
                costs = np.append(costs, cost)

                self.Visualize.Line.TreeArray(E)

                goal_found = self.is_goal(x_new, goal)
                if goal_found:
                    goal_data = np.asarray([goal_x, goal_y])
                    path = self.find_path(E, V, goal_data, car_point)
                    self.PurePursuit.Drive(path, car_point, angle_z)

    def sample(self, sample_space):
        number_of_rows = sample_space.shape[0]
        random_index = np.random.choice(number_of_rows, size=1, replace=False)
        random_row = sample_space[random_index][0]

        x_index = random_row[0]
        y_index = random_row[1]

        x = self.transform_grid[x_index][y_index][0]
        y = self.transform_grid[x_index][y_index][1]

        self.Visualize.Point.Sample(x, y)

        return random_row

    def nearest(self, V, sample_point):
        magnitudes = np.asarray([LA.norm(V - sample_point, axis = 1)])
        nearest_index = np.argmin(magnitudes)
        nearest_node = V[nearest_index]

        return nearest_node, nearest_index

    def steer(self, sample, near, node_length):
        x = sample[0] - near[0]
        y = sample[1] - near[1]

        angle = math.atan2(y, x)

        new_x = node_length * math.cos(angle) + near[0]
        new_y = node_length * math.sin(angle) + near[1]

        new_x_index = int(round(new_x))
        new_y_index = int(round(new_y))

        new_node = np.asarray([new_x_index, new_y_index])

        return new_node

    def check_collision(self, new_node, nearest_node):
        new_x = new_node[0]
        new_y = new_node[1]

        if self.occ_grid[new_x][new_y] == 1:
            return True

        near_x = nearest_node[0]
        near_y = nearest_node[1]

        line_coords = np.transpose(line(near_x, near_y, new_x, new_y))

        collides = line_coords[self.occ_grid[line_coords[:,0],line_coords[:,1]] == 1]          

        if len(collides) == 0:
            return False
        else:
            return True 

    def is_goal(self, x_new, goal):

        dist = LA.norm(x_new - goal)

        if dist < self.goal_threshold:
            return True
        else:
            return False

    def find_path(self, E, V, goal_data, car_point):
        V = self.transform_grid[V[:,0], V[:,1]]
        path = np.empty((0,2))
        path = np.vstack((path, goal_data))
        parent = len(V)-1
        root = E[parent][3]

        while not root:
            waypoint = V[parent]
            path = np.vstack((waypoint, path))
            parent = E[parent][2]
            root = E[parent][3]

        path = np.vstack((car_point, path))
        self.Visualize.Line.PathArray(path)

        return path
    
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

        goal_x, goal_y = self.localize_car_vectorized(xi, yi)

        self.Visualize.Point.Goal(xi, yi)
        return goal_x, goal_y, xi, yi

    def UpdateOccupancyGrid(self, car_point, car_index, angle_z):
        car_x = car_point[0]
        car_y = car_point[1]
        
        laser_x_coords, laser_y_coords = self.transform_lasers(self.distances, self.angles, car_x, car_y, angle_z) #0.0003
        laser_x_indices, laser_y_indices = self.localize_scan(laser_x_coords,laser_y_coords) 
        laser_indices = np.unique(np.column_stack((laser_x_indices, laser_y_indices)), axis=0) #0.0002

        car_x_index = car_index[0]
        car_y_index = car_index[1]

        self.scan_line_clear_vectorized(laser_indices[:,0], laser_indices[:,1], car_x_index, car_y_index) #0.0015

        #laser_indices = laser_indices[self.distances != self.r]

        self.scan_buffer_brens_vectorized(laser_indices[:,0], laser_indices[:,1])

    # The following methods are needed for RRT* and not RRT
    def cost(self, indices, node, E, V, costs):
        neighbor_data = E[indices]
        neighbor_points = V[indices]

        magnitudes = LA.norm(neighbor_points - node, axis =1)
        costs = costs[indices]
        total_costs = np.add(costs, magnitudes)

        best_neighbor_index = np.argmin(total_costs)
        best_neighbor = int(indices[best_neighbor_index])

        best_neighbor_cost = total_costs[best_neighbor_index]
        #print(type(best_neighbor))
        return best_neighbor, best_neighbor_cost

    def near_check(self, neighbor_x, neighbor_y, new_x, new_y):
        line_coords = np.transpose(line(neighbor_x, neighbor_y, new_x, new_y))
        collides = line_coords[self.occ_grid[line_coords[:,0],line_coords[:,1]] == 1]

        if len(collides) == 0:
            return True
        else:
            return False 

    def near(self, V, node):
        neighborhood = np.copy(self.empty)
        neighborhood_indices = np.add(self.neighborhood_stamp, node)
        neighborhood[neighborhood_indices[:,0], neighborhood_indices[:,1]] = 1
        neighbor_indices = np.where((neighborhood[V[:,0], V[:,1]] == 1))[0]
        return neighbor_indices

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
        raster = np.transpose(raster)
        raster = np.flip(raster, 1)

        return raster

    def print_raster_to_terminal(self, raster):
        s = 0

        for j in range(len(raster)-1):
            list = []
            for i in range(0+s,306+s):
                list.append(raster[i][j])
            print(list)

    def ObtainCarState(self, pose_msg):
        car_x = pose_msg.pose.pose.position.x
        car_y = pose_msg.pose.pose.position.y

        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w

        angle_z = self.QuatToEuler(qx,qy,qz,qw)

        """Moves car reference point from rear to front"""
        #car_x = car_x + car_length*math.cos(angle_z)
        #car_y = car_y + car_length*math.sin(angle_z)

        car_point = np.asarray([car_x, car_y])

        return car_point, angle_z

    def QuatToEuler(self,x,y,z,w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        rad = (math.atan2(siny_cosp, cosy_cosp))
        return rad

    def AlignMap(self, occ_grid):
        x_sig_figs = len(str(map_offset_x))-2
        y_sig_figs = len(str(map_offset_y))-2
        indices = np.indices((width, height))

        x_points = indices[0] * map_scalar + map_offset_x
        x_points_rounded = np.around(x_points, x_sig_figs)
        y_points = indices[1] * map_scalar + map_offset_y
        y_points_rounded = np.around(y_points, y_sig_figs)

        transform_grid = np.stack((x_points_rounded, y_points_rounded), axis=2)

        return transform_grid

    def AddBuffertoMap(self, occ_grid, safe_set):
        unsafe_set = []
        sample_set = np.empty((0,2))
        #print(safe_set)
        for point in safe_set:
            sum_ = 0
            buffer = np.add(self.scan_buffer, point)
            buffer = buffer.astype("int")
            for j in buffer:
                if (0 <= j[0] < width) and (0 <= j[1] < height) and (occ_grid[j[0]][j[1]] == 1):
                    sum_ += 1

            if sum_ > 0:
                unsafe_set.append(point)
            else:
                sample_set = np.vstack((sample_set, point))

        for i in unsafe_set:
            x = i[0]
            y = i[1]
            occ_grid[x][y] = 1

        sample_set = sample_set.astype("int")
        return occ_grid, sample_set

    #vectorized funtions
    def transform_lasers(self, distances, angles, car_x, car_y, angle_z):
        x = distances*np.cos(angles + angle_z) + car_x
        y = distances*np.sin(angles + angle_z) + car_y
        return x, y 

    def ScanLineClear(self, x_index, y_index, car_x_index, car_y_index):
        rr, cc = line(car_x_index, car_y_index, x_index, y_index)
        self.occ_grid[rr, cc] = 0

    def localize_car(self, x, y):
        x_index = int(round((x - map_offset_x)/map_scalar))
        y_index = int(round((y - map_offset_y)/map_scalar))
        
        return x_index, y_index

    def localize_scan(self, x, y):
        x_index = np.around((x - map_offset_x)/map_scalar).astype('int')
        y_index = np.around((y - map_offset_y)/map_scalar).astype('int')

        return x_index, y_index

    def scan_buffer_brens(self, laser_x, laser_y):
        laser_point = (laser_x, laser_y)
        t = np.add(self.scan_buffer, laser_point)
        self.occ_grid[t[:,0], t[:,1]] = 1

class PurePursuit(object):
    def __init__(self, transform_grid):
        global LOOKAHEAD
        global car_length
        global drive_topic

        self.L = LOOKAHEAD
        self.car_length = car_length
        self.s = rospy.get_param("/max_steer_angle")
        self.max_speed = rospy.get_param("/max_speed")
        self.min_speed = rospy.get_param("/min_speed")

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1)

        self.Visualize = Visualize(transform_grid)

    def FindNavIndex(self, distances, L):
        if L >= np.max(distances):
            return len(distances)-1
        differences = np.subtract(distances,L)
        i = np.where(differences > 0, differences, np.inf).argmin()
        return i

    def FindNavPoint(self, goal_index, magnitudes, waypoints, L):

        lower_index = goal_index - 1

        mi = 0 
        m1 = magnitudes[lower_index] - L
        m2 = magnitudes[goal_index] - L
        x1 = waypoints[lower_index][0]
        x2 = waypoints[goal_index][0]
        y1 = waypoints[lower_index][1]
        y2 = waypoints[goal_index][1]

        xi = np.interp(mi, [m1,m2], [x1, x2])
        yi = np.interp(mi, [m1,m2], [y1, y2])

        goal_point = np.asarray([xi,yi])
        return goal_point

    def Drive(self, waypoints, car_point, angle_z):
        L = self.L

        car_length = self.car_length

        magnitudes = LA.norm(waypoints - car_point, axis=1)

        goal_index = self.FindNavIndex(magnitudes, L)

        goal_point = self.FindNavPoint(goal_index, magnitudes, waypoints, L)
        
        x = (goal_point[0] - car_point[0])*math.cos(angle_z) + (goal_point[1] - car_point[1])*math.sin(angle_z)
        y = -(goal_point[0] - car_point[0])*math.sin(angle_z) + (goal_point[1] - car_point[1])*math.cos(angle_z)

        self.Visualize.Point.Nav(goal_point[0], goal_point[1])

        goal_for_car = np.asarray([x, y])
        d = LA.norm(goal_for_car)

        turn_radius = (d**2)/(2*(goal_for_car[1]))

        steering_angle = math.atan(car_length/turn_radius)

        s = self.s

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

        max_speed = self.max_speed
        min_speed = self.min_speed

        speed = (max_speed-min_speed)*(s - abs(steering_angle)) + min_speed

        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steering_angle
        self.ack.drive.speed = speed
        self.ack.header.stamp = rospy.Time.now()
        self.drive_pub.publish(self.ack)

class Visualize(object):
    def __init__(self, transform_grid):
        self.Point = self.Point()
        self.Line = self.Line(transform_grid)
        self.Array = self.Array(transform_grid)

    class Array(object):
        def __init__(self, transform_grid):
            self.occ_grid_pub = rospy.Publisher('/occupancy_grid_array', Marker, queue_size=1)
            self.wp_pub = rospy.Publisher('/waypoint_array', Marker, queue_size=1)
            self.scan_pub = rospy.Publisher('/scan_array', Marker, queue_size=1)
            self.buffer_pub = rospy.Publisher('/buffer_array', Marker, queue_size=1)
            self.clear_pub = rospy.Publisher('/clear_array', Marker, queue_size=1)
            self.past_pub = rospy.Publisher('/past_array', Marker, queue_size=1)
            self.sample_space_pub = rospy.Publisher('/sample_space_array', Marker, queue_size=1)

            self.compile_vectorized = np.vectorize(self.CompileArray)
            self.compile_transform_vectorized = np.vectorize(self.CompileTransform, otypes = [Point])

            self.transform_grid = np.copy(transform_grid)

        def CompileArray(self, x_index, y_index, h):
            p = Point()
            p.x = x_index
            p.y = y_index
            p.z = h
            return p

        def CompileTransform(self, x_index, y_index, h):
            p = Point()
            p.x = self.transform_grid[x_index][y_index][0]
            p.y = self.transform_grid[x_index][y_index][1]
            p.z = h
            return p
                
        def OccupancyGrid(self, sample_set_list):
            x_index_list = sample_set_list[:,0] #.astype('float')
            y_index_list = sample_set_list[:,1] #.astype('float')
            h = 0.01
            occ_points = self.compile_transform_vectorized(x_index_list, y_index_list, h)

            marker = Marker()
            marker.ns = "occ_grid"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = occ_points
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

            """sleep allows occupancy grid time to publish to RVIZ. it wont show up without this"""
            rospy.sleep(0.2)

            self.occ_grid_pub.publish(marker)

        def WaypointArray(self, waypoints):
            x_indices = waypoints[:,0]
            y_indices = waypoints[:,1]
            h = 0.01
            points = self.CompileArray(x_indices, y_indices, h)          

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

        def Scan(self, scan_x, scan_y):
            #print(scan_points)
            h = 0.02
            points = self.compile_vectorized(scan_x, scan_y, h)

            marker = Marker()
            marker.ns = "scan"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 0
            marker.color.r = 0
            marker.color.b = 1.0
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration(1)

            self.scan_pub.publish(marker)

        def Buffer(self, x_indices, y_indices):
            h = 0.05
            points = self.compile_transform_vectorized(x_indices, y_indices, h)         

            marker = Marker()
            marker.ns = "scan"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 0
            marker.color.r = 0
            marker.color.b = 0
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            #marker.lifetime = rospy.Duration(1)

            self.buffer_pub.publish(marker)

        def Clear(self, x_indices, y_indices):
            h = 0.03
            points = self.compile_transform_vectorized(x_indices, y_indices, h)           

            marker = Marker()
            marker.ns = "scan"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 1
            marker.color.r = 1
            marker.color.b = 1
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            #marker.lifetime = rospy.Duration(1)

            self.clear_pub.publish(marker)

        def Past(self, x_indices, y_indices):
            h = 0.04
            points = self.compile_transform_vectorized(x_indices, y_indices, h)           

            marker = Marker()
            marker.ns = "scan"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 1
            marker.color.r = 1
            marker.color.b = 1
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            #marker.lifetime = rospy.Duration(1)

            self.past_pub.publish(marker)

        def Sample(self, x_indices, y_indices):
            h = 0.05
            points = self.compile_transform_vectorized(x_indices, y_indices, h)           

            marker = Marker()
            marker.ns = "scan"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.POINTS
            marker.points = points
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.color.g = 0
            marker.color.r = 1
            marker.color.b = 1
            marker.color.a = 1
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            #marker.lifetime = rospy.Duration(1)

            self.sample_space_pub.publish(marker)

    class Point(object):
        def __init__(self):
            self.sample_pub = rospy.Publisher('/sample_point_marker', Marker, queue_size=1)
            self.car_pub = rospy.Publisher('/car_point_marker', Marker, queue_size=1)
            self.steer_pub = rospy.Publisher('/steer_point_marker', Marker, queue_size=1)
            self.goal_pub = rospy.Publisher('/goal_point_marker', Marker, queue_size=1)
            self.nav_pub = rospy.Publisher('/navigation_point_marker', Marker, queue_size=1)

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

        def Car(self, point):
            namespace = "car"
            green = 1
            red = 1
            blue = 0

            x = point[0]
            y = point[1]

            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.car_pub.publish(marker)

        def Steer(self, x, y):
            namespace = "steer"
            red = 0
            green = 0
            blue = 1

            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.steer_pub.publish(marker)

        def Goal(self, x, y):
            namespace = "goal"
            red = 1
            green = 0
            blue = 1

            marker = self.CreatePointMarker(namespace, x, y, red, green, blue)
            self.goal_pub.publish(marker)

    class Line(object):
        def __init__(self, transform_grid):
            self.tree_pub = rospy.Publisher('/tree_viz_array', Marker, queue_size=1)
            self.path_pub = rospy.Publisher('/path_viz_array', Marker, queue_size=1)

            self.transform_grid = transform_grid

            self.construct_list_points_vectorized = np.vectorize(self.ConstructListPoints)

        def ConstructListPoints(self, x_index, y_index, parent_index):
            if parent_index is not None:
                z = 0.025

                parent_x = self.E[parent_index][0]
                parent_y = self.E[parent_index][1]

                node_x = self.transform_grid[x_index][y_index][0]
                node_y = self.transform_grid[x_index][y_index][1]

                pnode_x = self.transform_grid[parent_x][parent_y][0]
                pnode_y = self.transform_grid[parent_x][parent_y][1]

                p1 = Point()
                p1.x = node_x
                p1.y = node_y
                p1.z = z

                p2 = Point()
                p2.x = pnode_x
                p2.y = pnode_y
                p2.z = z

                return p1, p2

        def TreeArray(self, E):
            self.E = E
            #print(E)
            x_indices = E[:,0]
            y_indices = E[:,1]
            parents = E[:,2]
            points = self.construct_list_points_vectorized(x_indices, y_indices, parents)

            points = points[points != None]


            points = list(sum(points, ()))

            marker = Marker()
            marker.ns = "tree"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.LINE_LIST
            marker.points = points
            marker.scale.x = 0.05
            marker.color.g = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            self.tree_pub.publish(marker)

        def PathArray(self, path):
            marker = Marker()
            points = []

            for i in range(len(path[:,0])-1):
                p = Point()
                p.x = path[i][0]
                p.y = path[i][1]
                p.z = 0.05

                points.append(p)

            #print(p)

            marker.ns = "path"
            marker.action = marker.MODIFY
            marker.header.frame_id = "/map"
            marker.type = marker.LINE_STRIP
            marker.points = points
            marker.scale.x = 0.05
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            self.path_pub.publish(marker)

def main():
    rospy.init_node('rrt')
    rrt = RRT()
    rospy.spin()

if __name__ == '__main__':
    main()