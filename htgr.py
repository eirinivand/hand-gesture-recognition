from __future__ import print_function
from flask_opencv_streamer.streamer import Streamer
import cv2
import sys
import base64
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
        frame = cv.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def video_stream(data_uri):
    img = data_uri_to_cv2_img(data_uri)
