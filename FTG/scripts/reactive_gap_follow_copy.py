#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class reactive_follow_gap:
    def __init__(self):
        #Topics & Subscriptions,Publishers
        lidarscan_topic = '/scan'
        drive_topic = '/nav'

        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidar_callback) #TODO
        self.drive_pub = rospy.Publisher('drive', AckermannDriveStamped, queue_size=1) #TODO
    
    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = list(ranges)
        #l = len(proc_ranges)
        #for index in range(0,l,10):
        #    mean = sum(proc_ranges[index:index+10])/10
        #    #if mean > 3:
        #    #    mean = float('inf')
        #    for x in range(index,index+10):
        #        proc_ranges[x] = mean

        #for index in range(0,269):
        #    proc_ranges[index]=0
        #for index in range(811,1080):
        #    proc_ranges[index]=0

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        return None
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
        return None

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        l = len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)

        #Find closest point to LiDAR
        l = len(data.ranges)
        min_val = float('inf')
        min_index = 0

        for index in range(0,l):
            if proc_ranges[index] < min_val:
                min_val = proc_ranges[index]
                min_index = index

        if min_index - 200 < 0:
            low = 0
        else:
            low = min_index -200

        if min_index + 200 > l:
            high = l
        else:
            high = min_index + 200

        for index in range(low, high):
            proc_ranges[index] = 0
            



        #Eliminate all points inside 'bubble' (set them to zero) 

        #Find max length gap 
        #start_index = 0

        #max_width = 0
        #for index in range(0,l):
        #    dist = proc_ranges[index]
        #    if index == 0:
        #        prev_dist = dist
        #    else:
        #        prev_dist = proc_ranges[index - 1]

        #    if (not np.isinf(prev_dist)) and (np.isinf(dist)):
        #        start_index = index
        #    elif (np.isinf(prev_dist)) and (not np.isinf(dist)):
        #        end_index = index
        #        width = end_index - start_index
        #        if width > max_width:
        #            max_width = width
        #    else:
        #        pass

        #Find the best point in the gap 
        best_index = 0
        best_val = 0
        
        for index in range(270,810):
            if proc_ranges[index] > best_val:
                best_val = proc_ranges[index]
                best_index = index

        #best_index = (start_index + end_index)/2
        #rospy.loginfo(str('start' + str(start_index)))
        #rospy.loginfo(str('end' + str(end_index)))
        #rospy.loginfo(str('Best' + str(best_index)))
        steer_angle = (data.angle_min + data.angle_increment * best_index)

        #Publish Drive message
        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steer_angle

        if best_val > 5 and (405 < best_index < 675):
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
