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
        #print(avg)
        final = np.multiply(NZ, 0)
        final = np.add(final,1)
        final = np.multiply(final, avg)
        final = np.concatenate(final)
        final = np.where(final == 100000, np.inf, final)

        proc_ranges = final

        #rospy.loginfo(proc_ranges)

        return proc_ranges

        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

    def bubble(self, proc_ranges, closest_index, inc):
        radius = car_width
        center = proc_ranges[closest_index]
        #print(center)
        theta = abs(math.atan2(radius,center))
        index_width = int(theta/inc)
        #print(index_width)
        bubble_max_index = closest_index + index_width
        #print(bubble_max_index)
        bubble_min_index = closest_index - index_width
        #print(bubble_min_index)

        indices = np.arange(bubble_min_index,bubble_max_index)
        #print(indices)
        np.put(proc_ranges,indices,0,mode='clip')

        return proc_ranges


    def find_max_gap(self, proc_ranges, inc):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        proc_ranges = np.where((proc_ranges == 0)|(proc_ranges == np.inf), 0, proc_ranges)
        #rospy.loginfo(proc_ranges)
        proc_ranges = np.split(proc_ranges, (np.where(abs(np.ediff1d(proc_ranges)) > 0)[0]+1))
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
        
        #print(widths)
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
        print(indexes)
        #rospy.loginfo(proc_ranges)
        #rospy.loginfo(indexes)
        return indexes
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
        gap_ranges = ranges[start_i:end_i]
        avg = np.mean(gap_ranges)
        indexes = np.subtract(gap_ranges, avg)
        indexes = np.abs(indexes)
        index = np.argmin(indexes) 
        #print(index)
        best_index = index + start_i
        

        #indices = np.arange(start_i,end_i)
        #best_index = np.median(indices)

        #indices = np.arange(start_i,end_i)
        #np.take(ranges,indices)
        #best_index = np.argmax(ranges) + start_i
        print(best_index)
        return best_index

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

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

        #Find max length gap 
        gap_indices = self.find_max_gap(proc_ranges, angle_inc)
        #rospy.loginfo(gap_indices)
        start_i = gap_indices[0]
        end_i = gap_indices[1]

        #Find the best point in the gap 
        best_index = self.find_best_point(start_i, end_i, ranges)
        #print(best_index)
        steer_angle = (data.angle_min + angle_inc * (best_index + min_index))

        #Publish Drive message
        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steer_angle
        #print(steer_angle)

        if (404 < best_index < 674):
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
