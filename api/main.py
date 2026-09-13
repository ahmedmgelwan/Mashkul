from fastapi import FastAPI
from src.config import model_info, PathConfig, DEVICE
from src.infer import Diacritizer
from contextlib import asynccontextmanager
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the diacritizer
    diacritizer = Diacritizer.from_pretrained(
        model_path=PathConfig.model_path,
        tokenizer_path=PathConfig.tokenizer_path,
        device=DEVICE
    )
    app.state.diacritizer = diacritizer
    yield
  

app = FastAPI(title=f'{model_info.model_name} API',
                         lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": model_info.model_name,
        "model_version": model_info.model_version,
        "model_description": model_info.model_description,
    } 

class DiacritizationRequest(BaseModel):
    text: str

class DiacritizationResponse(BaseModel):
    diacritized_text: str

@app.post("/diacritize", response_model=DiacritizationResponse)
def diacritize(request: DiacritizationRequest):
    diacritizer: Diacritizer = app.state.diacritizer
    diacritized_text = diacritizer.diacritize(request.text)
    return DiacritizationResponse(diacritized_text=diacritized_text)