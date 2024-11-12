
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import urllib.request
from keras_facenet import FaceNet

# Load mediapipe models
mp_face_detection = mp.solutions.face_detection 
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.8)

# Load FaceNet model
embedder = FaceNet()    

def url_to_image(image):
    resp = urllib.request.urlopen(image)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image