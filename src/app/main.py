from fastapi import FastAPI
from app.api.v1.face import router as face_router

app = FastAPI(
    title="My FastAPI App",
    description="A basic FastAPI application",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"Message": "Server FastAPI is running"}

app.include_router(face_router, prefix="/api/v1")
