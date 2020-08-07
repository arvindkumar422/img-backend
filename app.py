import cv2
import werkzeug
import urllib.request
import numpy as np
import gendetect
from flask_cors import CORS

import flask
from flask import request, jsonify, make_response


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


def faces(image):

    # imagePath = "https://www.thestatesman.com/wp-content/uploads/2017/08/1493458748-beauty-face-517.jpg";
    #
    # image = cv2.imread(imagePath)
    height, width = image.shape[:2]
    print("height:", height)
    print("width:", width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    )

    print("Found {0} Faces!".format(len(faces)))
    # print("rect: ", faces)

    res = []

    for (x, y, w, h) in faces:

        # rect = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        temp = {}

        temp['bottom_row'] = (y + h) / height
        temp['left_col'] = x / width
        temp['right_col'] = (x + w) / width
        temp['top_row'] = y / height

        res.append(temp)

        print(temp)
        # print("dims: ", x/width, y/height, (x+w)/width, (y+h)/height)
    #
    # status = cv2.imwrite('faces_detected.jpg', image)
    # print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

    return res


app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/getFaceRect', methods=['POST', 'OPTIONS'])
def home():
    try:
        if request.method == 'OPTIONS':
            print('options')
            return build_preflight_response()
        elif request.method == 'POST':
            imgurl = request.get_json()['imgUrl']
            image = url_to_image(imgurl)
            rect = faces(image)
            print("rect:", rect)
            if len(rect) == 0:
                rect = [{'bottom_row': 0.0, 'left_col': 0.0, 'right_col': 0.0, 'top_row': 0.0}]
            # print("response: ",build_actual_response(jsonify(rect)));
            return build_actual_response(jsonify(rect))
    except Exception as e:
        print(e)
        return jsonify({"error": "not found"})


@app.route('/getGeneralDetect', methods=['POST', 'OPTIONS'])
def general_detect():
    try:
        if request.method == 'OPTIONS':
            print('options')
            return build_preflight_response()
        elif request.method == 'POST':
            imgurl = request.get_json()['imgUrl']
            image = url_to_image(imgurl)
            res = gendetect.generalDetect(image)
            print("res: ", res)
            return build_actual_response(jsonify(res))
    except Exception as e:
        print(e)
        return jsonify({"error": "not found"})


@app.route('/', methods=['GET'])
def index():
    return "App is up and running (-:"


def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
