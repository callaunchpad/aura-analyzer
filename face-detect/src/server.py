# server.py
from flask import Flask, request, jsonify
from facedec import takeImage

app = Flask(__name__)

@app.route('/api/facedetect', methods=['POST'])
def face_detection(img_name = 'mskamalaharris.jpg'):
    # Get image path from the request (or you could send the actual image as form data)
    # data = request.json
    # image_path = data.get('image_path')

    # if not image_path:
    #     return jsonify({"error": "No image path provided"}), 400

    # Call the face detection function
    faceDetImg = takeImage(img_name)

    return jsonify({"faceDetImg": faceDetImg})

if __name__ == '__main__':
    app.run(debug=True)