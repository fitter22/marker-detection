import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

if len(sys.argv) < 2:
    print("Please, provide at least two arguments. Paths for marker and analyzed scene files.")
    sys.exit()

img1 = cv2.imread(str(sys.argv[1]), 0)  # marker image
img2 = cv2.imread(str(sys.argv[2]), 0)  # scene image

# Initiate detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

# store all the good matches as per Lowe's ratio test - https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # find perspective transformation of the marker on the scene
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    h,w = img1.shape  # get size of marker
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)  # find the object

    # dst contains four points which are placed around marker (p0, p1, p2, p3)
    # Point = (x, y)
    # p0 = (dst[0][0][0], dst[0][0][1])
    # p1 = (dst[1][0][0], dst[1][0][1])
    # etc...
    mean_x1 = 0.5 * (dst[0][0][0] + dst[1][0][0])
    mean_x2 = 0.5 * (dst[2][0][0] + dst[3][0][0])
    mean_y1 = 0.5 * (dst[0][0][1] + dst[3][0][1])
    mean_y2 = 0.5 * (dst[1][0][1] + dst[2][0][1])
    position_x = (mean_x1 + mean_x2) / 2
    position_y = (mean_y1 + mean_y2) / 2
    print("Marker found. Position of the marker:\nX:%d\nY:%d\n" % (position_x, position_y))

    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0, 0), 3, cv2.LINE_AA)  # draw rectangle around marker
    img2 = cv2.circle(img2, (np.int(position_x), np.int(position_y)), 10, 0, 10, 6, 0)  # make circle in center of the marker
    img2 = cv2.putText(img2, str("X:%d Y:%d" % (position_x, position_y)), (np.int(position_x + 20), np.int(position_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 6)  # put text with marker coordinates

    plt.imshow(img2, 'gray'), plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    sys.exit()
