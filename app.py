from flask import Flask, redirect, url_for, render_template, Response, request

from visual_manipulation import grayscale

app = Flask(__name__)
import urllib3
import learnopencv


@app.route('/', methods=['GET'])
def hello_world():
    data = request.host

    def image_dimensions():
        http = urllib3.PoolManager()

    return render_template("mask.html")


@app.route("/process", methods=['POST', 'GET'])
def process():
    data = request.form['data']
    if data:
        return learnopencv.keypoint(data)
    else:
        return data


def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/camera-feed", methods=['POST', 'GET'])
def camera_feed():
    import htgr
    return Response(gen(learnopencv.VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return htgr.video_stream()


@app.route("/mask-feed", methods=['POST', 'GET'])
def mask_feed():
    import learnopencv
    return Response(learnopencv.keypoint(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return htgr.video_stream()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port='5000')
