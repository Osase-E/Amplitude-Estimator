# Standard imports
import cv2
import numpy as np
import math
import sys

# Variable Initialisation
left_max_amplitude = 0
left_max_frame = 0
left_max_coords = (0, 0)
right_max_amplitude = 0
right_max_frame = 0
right_max_coords = (0, 0)
count = 0
path = []
SENSOR_LENGTH = math.sqrt((24**2)+(36**2))

###############################################################################
# CHANGE THE DIRECTORY FOR THE VIDEO-------------------------------------------
if len(sys.argv) > 1:
    vidcap = cv2.VideoCapture(str(sys.argv[1]))
else:
    vidcap = cv2.VideoCapture('C:\\Users\\osase\\Desktop\\Uni Work\\Third Year Porject\\pendulum.mp4')

###############################################################################


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
    params.filterByColor = True
    params.blobColor = 0
    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def amplitude_to_distance(focal_length, distance_to_object, x_midpoint, amplitude,aspect_ratio_x, aspect_ratio_y):
    fov = 2 * math.atan((SENSOR_LENGTH/(2*focal_length)))
    horizontal_fov = fov * (aspect_ratio_x/(aspect_ratio_x+aspect_ratio_y))
    size = distance_to_object * math.tan((horizontal_fov/2))
    amp_distance = 1.28 * (size*amplitude)/x_midpoint
    print("The object has an amplitude of: " + "{:.2f}".format(amp_distance) + "cm")
    return amp_distance


success, image = vidcap.read()
detector = init_blob_detector()

dimensions = image.shape
mid_y, mid_x = dimensions[0]//2, dimensions[1]//2

while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1


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

left_max_amplitude = abs(left_max_amplitude)
if left_max_amplitude > right_max_amplitude:
    max_amplitude = left_max_amplitude
    max_frame = left_max_frame
else:
    max_amplitude = right_max_amplitude
    max_frame = right_max_frame

print(amplitude_to_distance(26, 57, mid_x, max_amplitude, 9, 16))

