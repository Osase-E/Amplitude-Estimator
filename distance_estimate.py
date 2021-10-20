# Standard imports
import cv2
import numpy as np
import math
import sys

# Variable Initialisation
left_max_amplitude = 0      #the maximum amplitude in the left direction of the object
left_max_frame = 0          #the frame number that corresponds to that amplitude
left_max_coords = (0, 0)    #the coordinates of the centre of the object in the video for that frame
right_max_amplitude = 0     #the maximum amplitude in the right direction of the object
right_max_frame = 0         #the frame number that corresponds to that amplitude
right_max_coords = (0, 0)   #the coordinates of the centre of the object in the video for that frame
count = 0                   #initialiser to store the number of frames in the video
path = []                   #array to store the keypoints of each frame - drawing it shows the path of the object
SENSOR_LENGTH = math.sqrt((24**2)+(36**2))  #constant of the sensor size of the camera - CAN BE CHANGED

###############################################################################
# The directory of the video to be processed
if len(sys.argv) > 1:
    vidcap = cv2.VideoCapture(str(sys.argv[1]))
else:
    vidcap = cv2.VideoCapture('')

###############################################################################

# Create a blob detector
# Current settings are for a small ball that acted as a pendulum for calibration
def init_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 2800
    params.maxArea = 25000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
#     params.filterByColor = True
#     params.blobColor = 0
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

# Calculates the amplitude real distances of the oscillation based on the camera's focal length, the distance of the camera
# from the object, the x-coordinate of the midpoint, the amplitude in pixel distance of the object, the aspect ratio used to
# record the video
def amplitude_to_distance(focal_length, distance_to_object, x_midpoint, amplitude,aspect_ratio_x, aspect_ratio_y):
    CROPPING_FACTOR = 1.28 # iPhone videos seem to be cropped by a factor of 1.28, so this scales the distance
    
    fov = 2 * math.atan((SENSOR_LENGTH/(2*focal_length)))
    horizontal_fov = fov * (aspect_ratio_x/(aspect_ratio_x+aspect_ratio_y))
    size = distance_to_object * math.tan((horizontal_fov/2))
    amp_distance = CROPPING_FACTOR * (size*amplitude)/x_midpoint
    
    print("The object has an amplitude of: " + "{:.2f}".format(amp_distance) + "cm")
    return amp_distance

# Split the video into frames and store as JPEG images
success, image = vidcap.read()
while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
    
# Initialise the blob detector
detector = init_blob_detector()


# Get the pixel dimensions of the video
dimensions = image.shape
mid_y, mid_x = dimensions[0]//2, dimensions[1]//2

# For all frames stored, threshold and find the object of interest
for i in range(count):
    im = cv2.imread("frame%d.jpg" % i, cv2.IMREAD_GRAYSCALE)
    dimensions = im.shape
    blurred = cv2.GaussianBlur(im, (7, 7), 0)
    output = blurred.copy()
    temp = []

    (T, threshInv) = cv2.threshold(blurred, 180, 255,cv2.THRESH_BINARY_INV)

    # Detect blobs.
    keypoints = detector.detect(threshInv)
    for j in keypoints:
        if (j.pt[1] > (mid_y//2)) and (j.pt[1] < (mid_y+(mid_y//4))):
            path.append(j)
            temp.append(j)
    keypoints = temp
    amplitude = keypoints[0].pt[0] - mid_x

    # Check if the amplitude is in the left direction or right direction
    # Find if it is greater than the current amplitude in the respective direction
    # if it is, then overwrite the current store with the respective data
    if amplitude > 0:
        if right_max_amplitude < amplitude:
            right_max_amplitude = amplitude
            right_max_frame = i
            right_max_coords = keypoints[0].pt
            im_with_keypoints = cv2.drawKeypoints(threshInv, keypoints,
                                                  np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite("right.jpg", im_with_keypoints)
    else:
        if left_max_amplitude > amplitude:
            left_max_amplitude = amplitude
            left_max_frame = i
            left_max_coords = keypoints[0].pt
            im_with_keypoints = cv2.drawKeypoints(threshInv, keypoints,
                                                  np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite("left.jpg", im_with_keypoints)

    #Show the path of the pendulum
    if i == (count-1):
        im_with_keypoints = cv2.drawKeypoints(threshInv, path,
                                              np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("pendulum_path.jpg", im_with_keypoints)

# Find the absolute maximum amplitude, and the corresponding frame
left_max_amplitude = abs(left_max_amplitude)
if left_max_amplitude > right_max_amplitude:
    max_amplitude = left_max_amplitude
    max_frame = left_max_frame
else:
    max_amplitude = right_max_amplitude
    max_frame = right_max_frame

# Prints the real distance amplitude of the object
print(amplitude_to_distance(26, 57, mid_x, max_amplitude, 9, 16))

