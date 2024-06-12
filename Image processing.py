#!/usr/bin/env python3

# Import necessary Python libraries
import sys
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage

class DuckiebotLaneDetector:
    def __init__(self):
        self.bridge = CvBridge()

        # Initialize the ROS node and set up the subscriber for the image topic
        rospy.init_node('duckiebot_lane_detector')
        self.image_subscriber = rospy.Subscriber('/duckiebot/camera_node/image/compressed', CompressedImage, self.process_image, queue_size=1)

    def process_image(self, message):
        rospy.loginfo("Processing image...")

        # Convert the compressed image message to an OpenCV image
        image = self.bridge.compressed_imgmsg_to_cv2(message, "bgr8")

        # Set parameters for cropping the image
        crop_top = 200
        crop_bottom = 400
        crop_left = 100
        crop_right = 500

        # Crop the image based on the defined parameters
        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]

        # Convert the cropped image to the HSV color space
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # Define HSV range for white and yellow colors
        white_lower_bound = np.array([0, 0, 200])
        white_upper_bound = np.array([255, 50, 255])
        yellow_lower_bound = np.array([20, 100, 100])
        yellow_upper_bound = np.array([40, 255, 255])

        # Create masks for white and yellow colors
        white_mask = cv2.inRange(hsv_image, white_lower_bound, white_upper_bound)
        yellow_mask = cv2.inRange(hsv_image, yellow_lower_bound, yellow_upper_bound)

        # Detect edges in the masked images using the Canny Edge Detector
        white_edges = cv2.Canny(white_mask, 50, 150)
        yellow_edges = cv2.Canny(yellow_mask, 50, 150)

        # Perform Hough Transform to detect lines in the edge-detected images
        white_lines = self.hough_transform(white_edges)
        yellow_lines = self.hough_transform(yellow_edges)

        # Draw the detected lines on the cropped image
        self.render_lines(cropped_image, white_lines)
        self.render_lines(cropped_image, yellow_lines)

        # Display the image with the detected lane lines
        cv2.imshow('Lane Detection', cropped_image)
        cv2.waitKey(1)

    def hough_transform(self, edge_image):
        # Apply the Hough Line Transform
        lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
        return lines

    def render_lines(self, image, lines):
        # Draw the detected lines on the image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        lane_detector = DuckiebotLaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
