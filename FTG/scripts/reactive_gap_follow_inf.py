#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

#Global Values
first_iteration = True
min_index = 269 #269 = ~90 degrees to the right
max_index = 809 #809 = ~90 degrees to the left
car_width = 0.3 #meters
max_range = 6


class reactive_follow_gap:
    def __init__(self):
        #Topics & Subscriptions,Publishers
        lidarscan_topic = '/scan'
        drive_topic = '/nav'

        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidar_callback) #TODO
        self.drive_pub = rospy.Publisher('drive', AckermannDriveStamped, queue_size=1) #TODO
    
    def preprocess_lidar(self, ranges):
        NZ = np.where((ranges > max_range)|(ranges == np.isnan)|(ranges == np.isinf), 100000, ranges)
        NZ = np.split(NZ, np.where(abs(np.ediff1d(NZ)) > 10000)[0]+1)
        avg = [np.mean(i) for i in NZ]
        final = np.multiply(NZ, 0)
        final = np.add(final,1)
        final = np.multiply(final, avg)
        final = np.concatenate(final)
        final = np.where(final == 100000, np.inf, final)

        proc_ranges = final

        return proc_ranges

        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

    def bubble(self, proc_ranges, closest_index, inc):
        radius = car_width
        center = proc_ranges[closest_index]
        theta = math.atan2(radius,center)
        index_width = int(theta/inc)
        max_index = closest_index + index_width
        min_index = closest_index - index_width
        indices = np.arange(min_index,max_index)
        np.put(proc_ranges,indices,0,mode='clip')

        return proc_ranges


    def find_max_gap(self, proc_ranges, inc):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        proc_ranges = np.where((proc_ranges == 0)|(proc_ranges == np.isinf), 100000, proc_ranges)
        #rospy.loginfo(proc_ranges)
        proc_ranges = np.split(proc_ranges, (np.where(abs(np.ediff1d(proc_ranges)) > 10000)[0]+1))
        rows = 0

        for i in proc_ranges:
            rows = rows + 1
        angular_diameters = np.empty(rows)
        distances = np.empty(rows)

        for i in range(rows):
            row = proc_ranges[i]
            row_distance = row[0]
            distances = np.append(distances,row_distance)
            row_diameter = len(row)
            angular_diameters = np.append(angular_diameters,row_diameter)
        delete = np.arange(0,rows)
        angular_diameters = np.delete(angular_diameters,delete)
        distances = np.delete(distances,delete)
        distances = np.where(distances > 10000, 0, distances)

        widths = np.empty(rows)
        for i in range(rows):
            ang = inc*angular_diameters[i]
            distance = distances[i]
            gap_width = 2*distance*math.tan(ang/2)
            widths = np.append(widths, gap_width)

        widths = np.delete(widths,delete)

        max_gap_row = np.argmax(widths)

        max_gap = proc_ranges[max_gap_row]

        row_start_index = 0
        row_end_index = len(max_gap)-1
        replacement_value = max_gap[row_start_index]
        max_gap[row_start_index] = 101010101
        max_gap[row_end_index] = 909090909

        proc_ranges[max_gap_row] = max_gap
        proc_ranges = np.concatenate(proc_ranges)

        start_index = int(np.where(proc_ranges == 101010101)[0][0])
        end_index = int(np.where(proc_ranges == 909090909)[0][0])
        indexes = [start_index,end_index]
        return indexes
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
        indices = np.arange(start_i,end_i)
        np.take(ranges,indices,mode='clip')
        avg = np.mean(ranges)
        best_index = (np.abs(ranges-avg)).argmin()+ start_i


        #best_index = int((start_i+end_i)/2)

        #indices = np.arange(start_i,end_i)
        #np.take(ranges,indices)
        #best_index = np.argmax(ranges) + start_i
        return best_index

    def inf_gap_check(self, proc_ranges):
        proc_ranges = np.where((proc_ranges == np.isinf), 100000, proc_ranges)
        proc_ranges = np.split(proc_ranges, (np.where(abs(np.ediff1d(proc_ranges)) > 10000)[0]+1))
        avg = [np.mean(i) for i in proc_ranges]
        #print(avg)
        occurrances = 0
        for x in range(len(avg)-1):
            if avg[x] == np.inf:
                occurrances = occurrances + 1
        return occurrances

    def find_optimal_index(self, proc_ranges, prev_ranges, ranges, angle_inc):
        occurrances = self.inf_gap_check(proc_ranges)
        #rospy.loginfo("occur: " + str(occurrances))
        prev_occurrances = self.inf_gap_check(prev_ranges)
        #rospy.loginfo("prev_occur: " + str(prev_occurrances))
        if occurrances == 1 and prev_occurrances == 1:
            best_index = self.inf_gap_follow(proc_ranges, prev_ranges, angle_inc)
            prev_ranges = proc_ranges
            return best_index
        else:
            gap_indices = self.find_max_gap(proc_ranges, angle_inc)
            start_i = gap_indices[0]
            end_i = gap_indices[1]
            prev_ranges = proc_ranges
            best_index = self.find_best_point(start_i, end_i, ranges)
            return best_index

    def inf_gap_follow(self, proc_ranges, prev_ranges, angle_inc):
        truth_table = []
        truth_table = np.asarray(truth_table)
        for x in range(len(proc_ranges)-1):
            if proc_ranges[x] == np.inf and prev_ranges[x] == np.inf:
                truth_table = np.append(truth_table, 1)
            else:
                truth_table = np.append(truth_table, 0)
        indices = np.argwhere(truth_table)
        diff = np.ediff1d(indices)
        consecutive = True
        for x in range(len(diff)-1):
            if abs(diff[x]) > 1:
                consecutive = False
        if consecutive == True:
            best_index = np.median(indices)
            return best_index
        else: 
            best_index = self.original_best_point(proc_ranges, angle_inc)
            return best_index

        

    def original_best_point(self, proc_ranges, angle_inc, ranges):
        gap_indices = self.find_max_gap(proc_ranges, angle_inc)
        start_i = gap_indices[0]
        end_i = gap_indices[1]
        best_index = self.find_best_point(start_i, end_i, ranges)
        return best_index


    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        global prev_ranges
        global first_iteration

        #pull important data
        angle_inc = data.angle_increment
        angles = np.arange(min_index,max_index,angle_inc)

        ranges = data.ranges
        ranges = np.asarray(ranges)
        ranges = ranges[min_index:max_index]
        closest_index = np.argmin(ranges)

        #preprocess lidar and find closest point to LiDAR
        proc_ranges = self.preprocess_lidar(ranges)
        #rospy.loginfo(proc_ranges)
        
        #Eliminate all points inside 'bubble' (set them to zero) 
        proc_ranges = self.bubble(proc_ranges,closest_index,angle_inc)
        #rospy.loginfo(proc_ranges)

        #Find max length gap and find best point in the gap

        first_iteration = first_iteration
        if first_iteration == True:
            best_index = self.original_best_point(proc_ranges, angle_inc, ranges)
            first_iteration = False
            prev_ranges = proc_ranges
        else:
            best_index = self.find_optimal_index(proc_ranges, prev_ranges, ranges, angle_inc)
            prev_ranges = proc_ranges

        steer_angle = (data.angle_min + angle_inc * (best_index + min_index))
        #rospy.loginfo(steer_angle)

        #Publish Drive message
        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steer_angle

        if (405 < best_index < 675):
            speed = 2
        else:
            speed = 1

        self.ack.drive.speed = speed
        self.ack.header.stamp = rospy.Time.now()
        self.drive_pub.publish(self.ack)

def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = reactive_follow_gap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
