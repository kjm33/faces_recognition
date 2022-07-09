import numpy as np
from retinaface import RetinaFace
import dlib

predictor_path = "dlib/models/shape_predictor_5_face_landmarks.dat"
sp = dlib.shape_predictor(predictor_path)

face_rec_model_path = "dlib/models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

img_path = "../images/original_photos/IMG_20190826_132600.jpg"
detected_faces_desc = RetinaFace.detect_faces(img_path)
img = dlib.load_rgb_image(img_path)


for face_details in detected_faces_desc.values():
    face_roi = dlib.rectangle(*face_details['facial_area'])
    shape = sp(img, face_roi)
    # the vector to the screen.
    face_descriptor_vec = facerec.compute_face_descriptor(img, shape)
    face_descriptor = np.array(face_descriptor_vec)
    print(face_descriptor)
