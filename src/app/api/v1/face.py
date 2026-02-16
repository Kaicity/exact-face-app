from fastapi import APIRouter, UploadFile, File
from app.services.face_embedding import extract_face_embedding

router = APIRouter(tags=["Face"])

@router.post("/extract-face-ID")
async def extract_face(file: UploadFile = File(...)):
    embedding = await extract_face_embedding(file)
    return {"embedding": embedding}
