import io
import cv2
import face_recognition
import numpy as np
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import googleapiclient.discovery

#pip install opencv-python
#pip install face_recognition
#pip install setuptools

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Lists to hold face encodings and names
known_face_encodings = []
known_face_names = []

# Set up your Google Photos API credentials
CLIENT_SECRET = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Get list of all media items in a specific folder
folder_id = 'change_this_to_your_folder_id' #'0B8QoVaYLLQn2ZG9JQ3pZT3ZuZ0k

# List of image mime types
image_mime_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/raw"]

# Function to authenticate and create a service
def create_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
    creds = flow.run_local_server(port=8080)
    return googleapiclient.discovery.build(API_NAME, API_VERSION, credentials=creds)

# Function to get list of all media items from a specific folder
def get_media_items(service, folder_id):
    # Use the list method to fetch files
    results = service.files().list(q=f"'{folder_id}' in parents", pageSize=100).execute()
    items = results.get('files', [])
    return items

# Create a service
service = create_service()

# Get list of all media items
media_items = get_media_items(service, folder_id)

# Loop over each media item
for item in media_items:
    # Check if the item is an image
    if item['mimeType'] in image_mime_types:
        # Get the file ID
        file_id = item['id']
        # Request the file
        request = service.files().get_media(fileId=file_id)
        # Use MediaIoBaseDownload to download the file
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        # Reset the file handle position
        fh.seek(0)
        # Load image directly into face recognition
        image = face_recognition.load_image_file(fh)
        # Get face encodings for the image
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            face_encoding = face_encodings[0]
            # Add face encoding to list
            known_face_encodings.append(face_encoding)
            # Add name to list. We use the filename as the name
            known_face_names.append(item['name'])
        else:
            print(f"No faces found in the image {item['name']}")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
