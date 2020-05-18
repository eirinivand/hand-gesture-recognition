from flask import Flask, redirect, url_for, render_template, Response

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/mask')
def mask():
    return render_template("mask.html")


@app.route("/camera-feed", methods=['POST', 'GET'])
def camera_feed():
    import htgr
    return Response(htgr.opencv_streamer(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return htgr.video_stream()


@app.route("/mask-feed", methods=['POST', 'GET'])
def mask_feed():
    import learnopencv
    return Response(learnopencv.keypoint(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return htgr.video_stream()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
