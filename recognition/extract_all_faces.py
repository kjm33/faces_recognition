import numpy as np
from retinaface import RetinaFace
from pathlib import Path
from PIL import Image
from crop import CustomCrop

out_dir = Path("../images/ungrouped_faces/")
image_paths = Path("../images/original_photos/").glob("*jpg")

faces_idx = 1

for img_path in image_paths:
    detected_faces_desc = RetinaFace.detect_faces(str(img_path))
    if type(detected_faces_desc) != dict:
        continue

    for face_details in detected_faces_desc.values():

        match_score = float(face_details['score'])
        if match_score < 0.95:
            continue

        face_roi = face_details['facial_area']
        try:
            cropped_face_img = CustomCrop(width=300, height=300, face_percent=80).crop(str(img_path), *face_roi)
        except Exception:
            continue

        im = Image.fromarray(cropped_face_img)
        face_path = out_dir / f"{faces_idx}.jpg"
        im.save(str(face_path))
        faces_idx += 1
