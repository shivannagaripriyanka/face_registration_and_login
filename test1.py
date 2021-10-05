import dlib
import numpy as np
import cv2


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

img1 = cv2.imread("priya.jpeg")
img2 = cv2.imread("priya test.jpeg")

img1_detection = detector(img1, 1)
img2_detection = detector(img2, 1)
print(len(img1_detection))

img1_shape = sp(img1,img1_detection[0])
img2_shape = sp(img2,img2_detection[0])

img1_aligned = dlib.get_face_chip(img1, img1_shape)
img2_aligned = dlib.get_face_chip(img2, img2_shape)

img1_representation = model.compute_face_descriptor(img1_aligned)
img2_representation = model.compute_face_descriptor(img2_aligned)

img1_representation = np.array(img1_representation)
img2_representation = np.array(img2_representation)
print(img1_representation)
print(img2_representation)

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

distance = findEuclideanDistance(img1_representation, img2_representation)
threshold = 0.6

if distance < threshold:
    print(distance)
    print("photos are of same person")
else:
    print(distance)
    print("photos are of different persons")