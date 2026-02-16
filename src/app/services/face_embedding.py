from deepface import DeepFace
import numpy as np
import cv2
from fastapi import UploadFile

async def extract_face_embedding(file: UploadFile):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = DeepFace.represent(img_path=img, model_name="Facenet")
    embedding = result[0]["embedding"]
    return embedding
