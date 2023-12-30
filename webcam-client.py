from flask import Flask, render_template, Response, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)

# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
# for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
# camera = cv2.VideoCapture(0)

# Directory containing images
image_dir = "images"
# Lists to hold face encodings and names
known_face_encodings = []
known_face_names = []

# Loop over each file in the directory
for filename in os.listdir(image_dir):
    # Only process files with .jpeg or .jpg extension
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".PNG"):
        # Load image
        image = face_recognition.load_image_file(os.path.join(image_dir, filename))
        # Get face encodings for the image
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            face_encoding = face_encodings[0]
            # Add face encoding to list
            known_face_encodings.append(face_encoding)
            # Add name to list. We remove the file extension to get the name
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No faces found in the image {filename}")

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index-client.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['frame']
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return jsonify({'status': 'success', 'names': face_names})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')