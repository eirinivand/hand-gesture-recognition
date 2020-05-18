from __future__ import print_function
from flask_opencv_streamer.streamer import Streamer
import cv2
import sys
import time
import numpy as np
import cv2 as cv
import argparse

from flask import Blueprint

bp = Blueprint('video', __name__, url_prefix='/video')

scaling_factor = 0.8


def opencv_streamer():
    port = 3030
    require_login = False
    streamer = Streamer(port, require_login)

    # Open video device 0
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()

        streamer.update_frame(frame)

        if not streamer.is_streaming:
            streamer.start_streaming()

        cv2.waitKey(30)


def video_stream():
    args = {"aglo": "KNN"}
    if args["aglo"] == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    if not capture.isOpened:
        print('Unable to open: ')
        exit(0)
    while True:
        ret, frame = capture.read()
        frame = cv.resize(frame, (800, 600))
        fgmask = backSub.apply(frame)
        fgmask = cv.resize(fgmask, (802, 602))
        if ret == True:
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # corners = cv.goodFeaturesToTrack(gray, maxCorners=9, qualityLevel=0.05,
            #                                   minDistance=25)
            # corners = np.float32(corners)
            # for item in corners:
            #     x, y = item[0]
            #     cv.circle(frame, (x, y), 5, 255, -1)
            # face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3,
            #                                            minNeighbors=5)
            # for (x, y, w, h) in face_rects:
            #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            frame3 = cv.imencode('.jpg', frame)[1].tobytes()
            # img = cv.resize(img, (704, 396))
            # img2 = np.hstack((frame, frame2))
            # img3 = cv.imencode('.jpg', img2)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n')
            time.sleep(0.1)
        else:
            break


def mask_stream():
    args = {"aglo": "KNN"}
    if args["aglo"] == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    if not capture.isOpened:
        print('Unable to open: ')
        exit(0)
    while True:
        ret, img = capture.read()
        fgmask = backSub.apply(img)
        if ret == True:
            frame = cv.imencode('.jpg', fgmask)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break


def detect_shape():
    while True:
        ret, img = capture.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = cv.bitwise_not(thresh)

        element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5))

        morph_img = thresh.copy()
        cv.morphologyEx(src=thresh, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img)

        contours, _ = cv.findContours(morph_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        areas = [cv.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)

        # bounding box (red)
        cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
        r = cv.boundingRect(cnt)
        cv.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)

        # min circle (green)
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(img, center, radius, (0, 255, 0), 2)

        # fit ellipse (blue)
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(img, ellipse, (255, 0, 0), 2)


# Extract all the contours from the image
def get_all_contours(img):
    ref_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(ref_gray, 0, 255, cv.THRESH_BINARY)
    # Find all the contours in the thresholded image. The values
    # for the second and third parameters are restricted to a
    # certain number of possible values.
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours


def hand_track():
    # Input image containing all the shapes
    while True:
        ret, img = capture.read()
        img_orig = np.copy(img)
        input_contours = get_all_contours(img)
        solidity_values = []
        # Compute solidity factors of all the contours
        for contour in input_contours:
            [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
            cv.line(img, (x - 1, y), (0, vx), (0, 255, 0), 2)
        # Clustering using KMeans
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10,
                    1.0)
        flags = cv.KMEANS_RANDOM_CENTERS
        solidity_values = np.array(solidity_values).reshape((len(solidity_values), 1)).astype('float32')
        compactness, labels, centers = cv.kmeans(solidity_values, 5, None, criteria, 10, flags)
        closest_class = np.argmin(centers)
        output_contours = []
        for i in solidity_values[labels == closest_class]:
            index = np.where(solidity_values == i)[0][0]
            output_contours.append(input_contours[index])
            cv.drawContours(img, output_contours, -1, (0, 0, 0), 3)
        # Censoring
        for contour in output_contours:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img_orig, [box], 0, (0, 0, 0), -1)
        frame = cv.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def draw_hand_0():
    while True:
        BLUE = (255, 0, 0)
        ret, img = capture.read()
        # Let's define four points
        line_points = [[(31, 5), (31, 22)],
                       [(31, 40), (31, 22)],
                       [(31, 57), (31, 40)],
                       [(55, 94), (31, 57)],
                       [(48, 56), (55, 94)],
                       [(48, 39), (48, 56)],
                       [(48, 22), (48, 39)],
                       [(48, 0), (48, 22)],
                       [(66, 4), (65, 22)],
                       [(66, 39), (65, 22)],
                       [(66, 56), (66, 39)],
                       [(55, 94), (66, 56)],
                       [(83, 57), (55, 94)],
                       [(83, 39), (83, 57)],
                       [(83, 22), (83, 39)],
                       [(83, 11), (83, 22)],
                       [(0, 44), (6, 53)],
                       [(16, 65), (6, 53)],
                       [(27, 80), (16, 65)],
                       [(55, 94), (27, 80)]]
        for line in line_points:
            cv.line(img, line[0], line[1], BLUE, 2)
        frame = cv.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


points = []
nPoints = 22
protoFile = "./files/hand/pose_deploy.prototxt"
weightsFile = "./files/hand/pose_iter_102000.caffemodel"


def draw_hand():
    # while True:
    strarttime = time.time()
    ret, frame = capture.read()
    frame = cv.resize(frame, (800, 600))
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    for i in range(nPoints):
        # confidence map of corresponding body's part.

        inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (500, 400),
                                       (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()
        probMap = output[0, i, :, :]
        probMap = cv.resize(probMap, (800, 600))
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
        frameCopy = frame.copy()
        if prob > 0.7:
            cv.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
            cv.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 2, lineType=cv.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
    # Draw Skeleton
    for pair in points:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
    endtime = time.time()
    print(strarttime - endtime)
    frame_encoded = cv.imencode('.jpg', frame)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')
