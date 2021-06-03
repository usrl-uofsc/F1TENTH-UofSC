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
min_index = 0 #269 = ~90 degrees to the right
max_index = 0 #809 = ~90 degrees to the left
deg_step = 0
velocity_coeff = 0

in_sim = False

car_width = 0.3 #meters
max_range = 6
LOOKAHEAD = 0.5

#pid values
k_gain = 0.5 
kp = 2
kd = 1
ki = 0.125
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0
angle_a = 30

class reactive_follow_gap:
    def __init__(self):
        #Topics & Subscriptions,Publishers
        global min_index
        global max_index
        global deg_step
        global in_sim
        global velocity_coeff
        lidarscan_topic = '/scan'

        if in_sim == True:
                min_index = 270
                max_index = 810
                deg_step = 3
                drive_topic = '/drive'
                velocity_coeff = 1
        else:
                min_index = 180
                max_index = 900
                deg_step = 4
                drive_topic = '/vesc/low_level/ackermann_cmd_mux/input/navigation'
                velocity_coeff = -1

        self.pid_iterations = 0

        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidar_callback) #TODO
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1) #TODO
    
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

        #rospy.loginfo(proc_ranges)

        return proc_ranges

        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

    def bubble(self, proc_ranges, closest_index):
        radius = car_width
        center = proc_ranges[closest_index]
        theta = math.atan2(radius,center)
        index_width = int(theta/self.angle_inc)
        bubble_max_index = closest_index + index_width
        bubble_min_index = closest_index - index_width
        indices = np.arange(bubble_min_index,bubble_max_index)
        np.put(proc_ranges,indices,0,mode='clip')

        #rospy.loginfo(proc_ranges)

        return proc_ranges


    def find_max_gap(self, proc_ranges):
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
            ang = self.angle_inc*angular_diameters[i]
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


        #best_index = int((start_i+end_i)/2)
        #indices = np.arange(start_i,end_i)
        #np.take(ranges,indices)
        #best_index = np.argmax(ranges) + start_i
        return best_index

    def PID(self, ranges):
        global integral
        global prev_error
	global k_gain
        global kp
        global ki
        global kd

        right_distance = self.followRight(ranges)
        left_distance = self.followLeft(ranges)
        error = (left_distance - right_distance)
    
        self.current_time = rospy.get_time()*1000
        delta_t = self.current_time - self.old_time
        self.old_time = self.current_time  
        total_time = self.current_time - self.start_time
        
        angle = float(kp*error*k_gain + (ki*k_gain*integral/total_time) + kd*k_gain*((error - prev_error)/delta_t)) #Use kp, ki & kd to implement a PID controller for 
        integral = (integral + error)
	#print(integral)
        prev_error = error

        return angle

    def steering_logic(self, steer_angle_from_FTG, ranges):
	#print(steer_angle_from_FTG)
        global prev_error 
        global error
        global integral
        global threshold_angle

        if self.pid_iterations == 0:
            integral = 0
            error = 0
            prev_error = 0
            self.start_time = rospy.get_time()*1000
            self.old_time = self.start_time

        steer_angle = self.PID(ranges)
        self.pid_iterations = self.pid_iterations + 1
	return steer_angle

        if abs(steer_angle) < abs(steer_angle_from_FTG):
            print("pid")
            return steer_angle
        else:
            print("FTG")
            self.pid_iterations = 0
            return steer_angle_from_FTG

    def followLeft(self, ranges):
        global angle_a
        global max_index
	global LOOKAHEAD
        global deg_step

	L = LOOKAHEAD
        b_inc = max_index
	#print("max index: " + str(ranges[b_inc]))
        a_inc = max_index - deg_step * angle_a
	#print("min index: " + str(ranges[a_inc]))
        start_ang = self.start_angle
        self.b_ang = start_ang+(self.angle_inc * b_inc)
	#print("b_angl: " + str(self.b_ang))
        self.b = ranges[b_inc]

        left_ranges = ranges[a_inc:b_inc]
	#print(left_ranges)
        left_angles = [self.b_ang - (self.angle_inc * i) for i in range(1,len(left_ranges))]
	np.flip(left_angles)
	#print(left_angles)
        left_thetas = [float((self.b_ang-i)) for i in left_angles]
	#print(left_thetas)
        left_alphas = [math.atan(((left_ranges[i] * math.cos(left_thetas[i]))-self.b)/(left_ranges[i]*math.sin(left_thetas[i]))) for i in range(len(left_ranges)-1)]
        #print(left_alphas)
        left_distances = [self.b * math.cos(left_alphas[i]) for i in range(len(left_ranges)-1)]
	#print(left_distances)
	left_lookahead = [left_distances[i]  + L*math.sin(left_alphas[i]) for i in range(len(left_ranges)-1)]
        left_lookahead[left_lookahead != np.inf]
        left_dist = np.mean(left_lookahead) 


	#print("left avg: " + str(left_dist))
        return left_dist

    def followRight(self,ranges):
        global angle_a
        global min_index
	global LOOKAHEAD
        global deg_step

	L = LOOKAHEAD
        b_inc = min_index
        a_inc = min_index + deg_step * angle_a
        start_ang = self.start_angle
        self.b = ranges[b_inc]
        self.b_ang = start_ang+(self.angle_inc * b_inc)
	#print("b_angr: " + str(self.b_ang))

        right_ranges = ranges[b_inc:a_inc]
	#print(right_ranges)
        right_angles = [(self.angle_inc * i) + self.b_ang for i in range(1, len(right_ranges))]
	#print(right_angles)
        right_thetas = [float((i-self.b_ang)) for i in right_angles]
	#print(right_thetas)
        right_alphas = [math.atan(((right_ranges[i] * math.cos(right_thetas[i]))-self.b)/(right_ranges[i]*math.sin(right_thetas[i]))) for i in range(len(right_ranges)-1)]

        right_distances = [self.b * math.cos(right_alphas[i]) for i in range(len(right_ranges)-1)]

	right_lookahead = [right_distances[i]  + L*math.sin(right_alphas[i]) for i in range(len(right_ranges)-1)]
        right_lookahead[right_lookahead != np.inf]
        right_dist = np.mean(right_lookahead) 

	#print("right avg: " + str(right_dist))
        return right_dist

    def FTG(self,proc_ranges,closest_index,ranges):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
   
       
        #Eliminate all points inside 'bubble' (set them to zero) 
        proc_ranges = self.bubble(proc_ranges,closest_index)

        #Find max length gap 
        gap_indices = self.find_max_gap(proc_ranges)
        #rospy.loginfo(gap_indices)
        start_i = gap_indices[0]
        end_i = gap_indices[1]

        #Find the best point in the gap 
        best_index = self.find_best_point(start_i, end_i, ranges)
        steer_angle_from_FTG = (self.start_angle + self.angle_inc * (best_index + min_index))

        #steer_angle = steer_angle_from_FTG
        return steer_angle_from_FTG
        

    def lidar_callback(self, data):
        ranges = data.ranges
        ranges = np.asarray(ranges)
        trimmed_ranges = ranges[min_index:max_index]

        self.angle_inc = data.angle_increment
        self.start_angle = data.angle_min
        closest_index = np.argmin(trimmed_ranges)
        
        #preprocess lidar and find closest point to LiDAR
        proc_ranges = self.preprocess_lidar(trimmed_ranges)

        steer_angle_from_FTG = self.FTG(proc_ranges,closest_index, data.ranges)
        steer_angle = self.steering_logic(steer_angle_from_FTG, data.ranges)


        #Publish Drive message
        self.ack = AckermannDriveStamped()
        self.ack.header.frame_id = 'steer'
        self.ack.drive.steering_angle = steer_angle

        if 0 <= abs(steer_angle) < 10:
            velocity = 1.5
        elif 10 <= abs(steer_angle) < 20:
            velocity = 1.0
        else:
            velocity = 0.5

        velocity = velocity_coeff*velocity

        self.ack.drive.speed = velocity
        self.ack.header.stamp = rospy.Time.now()
        self.drive_pub.publish(self.ack)

        #print(steer_angle)



def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = reactive_follow_gap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
