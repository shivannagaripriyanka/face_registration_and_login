import dlib
import cv2
import numpy as np
from scipy import misc
# Models Loaded
face_detector = dlib.get_frontal_face_detector()
pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
def whirldata_face_detectors(img, number_of_times_to_upsample=1):
 return face_detector(img, number_of_times_to_upsample)
def whirldata_face_encodings(face_image,num_jitters=1):
 face_locations = whirldata_face_detectors(face_image)
 pose_predictor = pose_predictor_5_point
 predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
 return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]
known_image = cv2.imread("priya.jpeg")
print(known_image.shape)
enc = whirldata_face_encodings(known_image)
print(enc)
print(type(enc))