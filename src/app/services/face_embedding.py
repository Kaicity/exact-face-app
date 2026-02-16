from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
import numpy as np
import cv2
from fastapi import UploadFile, HTTPException

async def extract_face_embedding(file: UploadFile):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.represent(img_path=img, model_name="Facenet", enforce_detection=True)
    except FaceNotDetected:
        raise HTTPException(
            status_code= 400,
            detail="Không phát hiện được khuôn mặt trong ảnh. Hãy dùng ảnh chân dung rõ hơn."
        )
    
    embedding = result[0]["embedding"]

    return embedding
