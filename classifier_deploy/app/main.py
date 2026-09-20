from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner

from app.config import settings
from app.database import engine, Base
from app import predict as predict_module
from app.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables and load model
    Base.metadata.create_all(bind=engine)

    model_path = Path(settings.models_dir) / "caltech_resnet50_cpu.pkl"
    print(f"Loading model from {model_path}...")
    predict_module.learn = load_learner(model_path, cpu=True)
    print(f"Model loaded. Classes: {predict_module.learn.dls.vocab}")

    yield


app = FastAPI(title="Caltech Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predict_module.learn is not None}
