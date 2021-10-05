from flask import Flask,request
import dlib, cv2
from flask_sqlalchemy import SQLAlchemy
import numpy as np


app=Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/db'
app.debug = True
db = SQLAlchemy(app)

class images(db.Model):
    __tablename__ = "facerecognition"
    Name = db.Column(db.String(), nullable=False, primary_key=True)
    Image = db.Column(db.String(), nullable=False)

def __init__(self,User_name,Imagedata):
    self.User_name = User_name
    self.Imagedata = Imagedata

@app.route("/registration",methods=["POST"])
def User_reg():
    name=request.values["User_Name"]
    image=request.files["Image"].read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    face_detector = dlib.get_frontal_face_detector()
    pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    def whirldata_face_detectors(img, number_of_times_to_upsample=1):
        return face_detector(img, number_of_times_to_upsample)
    def whirldata_face_encodings(face_image, num_jitters=1):
        face_locations = whirldata_face_detectors(face_image)
        pose_predictor = pose_predictor_5_point
        predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
        return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in
                predictors][0]
    enc = whirldata_face_encodings(image)
    print(enc)
    l=[]
    for i in enc:
        l.append(i)
    input=images(Name=name,Image=l)
    db.session.add(input)
    db.session.commit()
    return "Success"




@app.route("/login",methods=["POST"])
def login():
    image = request.files["Image"].read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    face_detector = dlib.get_frontal_face_detector()
    pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    def whirldata_face_detectors(img, number_of_times_to_upsample=1):
        return face_detector(img, number_of_times_to_upsample)

    def whirldata_face_encodings(face_image, num_jitters=1):
        face_locations = whirldata_face_detectors(face_image)
        pose_predictor = pose_predictor_5_point
        predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
        return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in
                predictors][0]
    loginenc = whirldata_face_encodings(image)
    print(loginenc)

    def findEuclideanDistance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    user_check = Faces.query.all()
    l1 = []
    l2 = []
    for i, j in enumerate(user_check):
        for k in j.Image.split(","):
            if "{" in k:
                k = k.replace("{", " ")
                k = float(k)
                l1.append(k)
            elif "}" in k:
                k = k.replace("}", " ")
                k = float(k)
                l1.append(k)
            elif "{" and "}" not in k:
                k = float(k)
                l1.append(k)
        l2.append(l1)
        l1 = []
    for e, f in enumerate(l2):
        print(type(l2[e]))
        array = np.array(f, dtype="float64")
        print(len(array))
    distance = findEuclideanDistance(array,loginenc)
    threshold = 0.6
    if distance < threshold:
        print(distance)
        return f"Authentication Successful"


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)