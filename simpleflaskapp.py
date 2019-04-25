from flask import Flask, render_template, Response, request, make_response
# from camera import Camera
from opencvcam import Camera
import cv2

app = Flask(__name__)

@app.route("/")
def sendHello():
    return render_template('index.html')

def cameraGen(camera):
    while True:
        img = camera.get_frame()
        # img = img.toBytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(cameraGen(Camera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)