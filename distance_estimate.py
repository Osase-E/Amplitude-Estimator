import cv2
import numpy as np
import math
import sys

# Variable Initialization
left_max_amplitude = 0
left_max_frame = 0
left_max_coords = (0, 0)
right_max_amplitude = 0
right_max_frame = 0
right_max_coords = (0, 0)
path = []
SENSOR_LENGTH = math.sqrt(24 ** 2 + 36 ** 2)

# The directory of the video to be processed
input_video = str(sys.argv[1]) if len(sys.argv) > 1 else ''
vidcap = cv2.VideoCapture(input_video)

# Create a blob detector
def init_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 2800
    params.maxArea = 25000
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

# Converts amplitude to distance
def amplitude_to_distance(focal_length, distance_to_object, x_midpoint, amplitude, aspect_ratio_x, aspect_ratio_y):
    CROPPING_FACTOR = 1.28
    fov = 2 * math.atan(SENSOR_LENGTH / (2 * focal_length))
    horizontal_fov = fov * (aspect_ratio_x / (aspect_ratio_x + aspect_ratio_y))
    size = distance_to_object * math.tan(horizontal_fov / 2)
    amp_distance = CROPPING_FACTOR * (size * amplitude) / x_midpoint
    print(f"The object has an amplitude of: {amp_distance:.2f} cm")
    return amp_distance

# Split the video into frames and store as JPEG images
count = 0
while True:
    success, image = vidcap.read()
    if not success:
        break
    cv2.imwrite(f"frame{count}.jpg", image)
    count += 1

# Initialize the blob detector
detector = init_blob_detector()

# Get the pixel dimensions of the video
dimensions = image.shape
mid_y, mid_x = dimensions[0] // 2, dimensions[1] // 2

# Process each frame
for i in range(count):
    im = cv2.imread(f"frame{i}.jpg", cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(im, (7, 7), 0)
    thresh_inv = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)[1]

    # Detect blobs
    keypoints = detector.detect(thresh_inv)
    temp = [j for j in keypoints if mid_y // 2 < j.pt[1] < mid_y + (mid_y // 4)]
    path.extend(temp)
    amplitude = temp[0].pt[0] - mid_x

    # Check amplitude direction and update max amplitudes
    if amplitude > 0 and right_max_amplitude < amplitude:
        right_max_amplitude = amplitude
        right_max_frame = i
        right_max_coords = temp[0].pt
        im_with_keypoints = cv2.drawKeypoints(thresh_inv, temp, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("right.jpg", im_with_keypoints)
    elif amplitude < 0 and left_max_amplitude > amplitude:
        left_max_amplitude = amplitude
        left_max_frame = i
        left_max_coords = temp[0].pt
        im_with_keypoints = cv2.drawKeypoints(thresh_inv, temp, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("left.jpg", im_with_keypoints)

    # Show the path of the pendulum
    if i == count - 1:
        im_with_keypoints = cv2.drawKeypoints(thresh_inv, path, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("pendulum_path.jpg", im_with_keypoints)

# Find the absolute maximum amplitude and corresponding frame
left_max_amplitude = abs(left_max_amplitude)
max_amplitude = max(left_max_amplitude, right_max_amplitude)
max_frame = left_max_frame if left_max_amplitude > right_max_amplitude else right_max_frame

# Print the real distance amplitude of the object
print(amplitude_to_distance(26, 57, mid_x, max_amplitude, 9, 16))
