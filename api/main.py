from fastapi import FastAPI
from src.config import model_info

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": model_info.model_name,
        "model_version": model_info.model_version,
        "model_description": model_info.model_description,
    } 