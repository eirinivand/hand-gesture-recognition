from __future__ import division
import cv2
import time
import numpy as np
import base64
from flask import Blueprint

from kaggle.api.kaggle_api_extended import KaggleApi

bp = Blueprint('bbb', __name__, url_prefix='/bbb')

api = KaggleApi()
api.authenticate()

api.dataset_download_file('changethetuneman/openpose-model', 'pose_iter_102000.caffemodel', path="files/hand/")

protoFile = "./files/hand/pose_deploy.prototxt"
weightsFile = "./files/hand/pose_iter_102000.caffemodel"

nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

threshold = 0.1

# defining face detector
face_cascade = cv2.CascadeClassifier("cascade_files/aGest.xml")
ds_factor = 0.6


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def from_b64(uri):
    '''
        Convert from b64 uri to OpenCV image
        Sample input: 'data:image/jpg;base64,/9j/4AAQSkZJR......'
    '''
    encoded_data = uri.split(',')[1]
    data = base64.b64decode(encoded_data)
    np_arr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def to_b64(img):
    '''
        Convert from OpenCV image to b64 uri
        Sample output: 'data:image/jpg;base64,/9j/4AAQSkZJR......'
    '''
    _, buffer = cv2.imencode('.jpg', img)
    uri = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpg;base64,{uri}'


def keypoint(base64_data):
    try:
        frame = from_b64(base64_data)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight
        t = time.time()
        # input image dimensions for the network
        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        print("time taken by network : {:.3f}".format(time.time() - t))
        return to_b64(frame)
    except:
        # just in case some process is failed
        # normally, for first connection
        # return the original data
        return base64_data


def keypoint_video():
    cap = cv2.VideoCapture(0)
    hasFrame, frame = cap.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth / frameHeight

    inHeight = 368
    inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    k = 0
    while True:
        k += 1
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        print("forward = {}".format(time.time() - t))

        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold:
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                            (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        print("Time Taken for frame = {}".format(time.time() - t))
        frame_encoded = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')
