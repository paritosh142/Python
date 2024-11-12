import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import urllib.request
import os
import logging
import tensorflow as tf
import absl.logging

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# Set TensorFlow logging to only show errors
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)
logging.disable(logging.WARNING)

mp_face_detection = mp.solutions.face_detection
embedder = FaceNet()
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def url_to_image(image):
    resp = urllib.request.urlopen(image)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Function to detect face and extract the bounding box
def detect_faces(image):
    results = face_detection.process(image)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            return face_resized
    return None

# Precompute embeddings for real-front, real-right, real-left
image_front = url_to_image("https://visionwayqa-qauseast1-bucket-common.s3.amazonaws.com/1332/lead-images/paritosh@centocode.com/front_face.jpg")
image_right = url_to_image("https://visionwayqa-qauseast1-bucket-common.s3.amazonaws.com/1332/lead-images/paritosh@centocode.com/right_face.jpg")
image_left = url_to_image("https://visionwayqa-qauseast1-bucket-common.s3.amazonaws.com/1332/lead-images/paritosh@centocode.com/left_face.jpg")

# companyid = "1332"
# email = "paritosh@centocode.com"

# image_front = url_to_image("https://visionwayqa-qauseast1-bucket-common.s3.amazonaws.com/{companyid}/lead-images/{email}/front_face.jpg")
# image_right = url_to_image("https://visionwayqa-qauseast1-bucket-common.s3.amazonaws.com/{companyid}/lead-images/{email}/right_face.jpg")
# image_left =  url_to_image("https://visionwayqa-qauseast1-bucket-common.s3.amazonaws.com/{companyid}/lead-images/{email}/left_face.jpg")

face_front = detect_faces(image_front)
face_right = detect_faces(image_right)
face_left =  detect_faces(image_left)

real_embeddings = embedder.embeddings([face_front, face_right, face_left])

# Function to calculate Euclidean distance
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Function to calculate Cosine Similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1.flatten(), embedding2.flatten())
    norm1 = np.linalg.norm(embedding1.flatten())
    norm2 = np.linalg.norm(embedding2.flatten())
    return dot_product / (norm1 * norm2)

# Start live video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_live = detect_faces(frame)
    
    if face_live is not None:
        live_embedding = embedder.embeddings([face_live])[0]

        # Calculate distances with pre-stored images
        distance_front = euclidean_distance(live_embedding, real_embeddings[0])
        distance_right = euclidean_distance(live_embedding, real_embeddings[1])
        distance_left = euclidean_distance(live_embedding, real_embeddings[2])

        # Calculate cosine similarity with pre-stored images
        cosine_front = cosine_similarity(live_embedding, real_embeddings[0])
        cosine_right = cosine_similarity(live_embedding, real_embeddings[1])
        cosine_left = cosine_similarity(live_embedding, real_embeddings[2])

        cv2.putText(frame, f"Distance : {distance_front:.2f}, {distance_right:.2f}, {distance_left:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cosine : {cosine_front:.2f}, {cosine_right:.2f}, {cosine_left:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Define thresholds for recognition
        if (distance_front< 0.8 or distance_right < 0.8 or distance_left < 0.8) and (cosine_right > 0.5 and cosine_left > 0.5 and cosine_front > 0.5):
            cv2.putText(frame, "Recognized", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Video", frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
