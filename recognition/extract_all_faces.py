import numpy as np
from retinaface import RetinaFace
from pathlib import Path
from PIL import Image

out_dir = Path("../images/ungrouped_faces/")
image_paths = Path("../images/original_photos/").glob("*jpg")

faces_idx = 1

for img_path in image_paths:
    faces = RetinaFace.extract_faces(img_path=str(img_path))
    for face_img in faces:
        im = Image.fromarray(face_img)
        face_path = out_dir / f"{faces_idx}.jpg"
        im.save(str(face_path))
        faces_idx += 1
