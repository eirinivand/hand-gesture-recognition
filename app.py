from flask import Flask, redirect, url_for, render_template, Response
from camera import VideoCamera

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/mask')
def mask():
    return render_template("mask.html")


import htgr, learnopencv

app.register_blueprint(htgr.bp)
app.add_url_rule('/', endpoint='video')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/camera-feed", methods=['POST', 'GET'])
def camera_feed():
    return Response(htgr.video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return htgr.video_stream()


@app.route("/mask-feed", methods=['POST', 'GET'])
def mask_feed():
    return Response(learnopencv.keypoint(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return htgr.video_stream()


if __name__ == '__main__':
    app.run()
