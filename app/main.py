"""
FastAPI Backend Application for Next Word Prediction.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.model import model_manager
from app.schemas import (
    PredictRequest,
    PredictResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelMetadata
)
from app.predictor import predict_next_words, generate_sequence


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML Model & Tokenizer
    model_path = os.getenv("MODEL_PATH", "model/next_word_model.keras")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "model/tokenizer.pkl")
    config_path = os.getenv("CONFIG_PATH", "model/config.json")
    
    print("[App Startup] Initializing ModelManager...")
    model_manager.load_resources(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        config_path=config_path
    )
    yield
    print("[App Shutdown] Cleaning up resources...")


app = FastAPI(
    title="Next Word Prediction API",
    description="Deep Learning Next Word Prediction & Autocomplete Engine with LSTM / Neural Language Modeling",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "model_not_ready",
        model_loaded=model_manager.is_loaded,
        version="1.0.0"
    )


@app.get("/api/metadata", response_model=ModelMetadata, tags=["Model"])
async def get_model_metadata():
    """Retrieve model configuration and vocabulary metadata."""
    return ModelMetadata(
        model_name="LSTM Next Word Predictor",
        status="active" if model_manager.is_loaded else "uninitialized",
        vocab_size=model_manager.vocab_size,
        max_sequence_length=model_manager.max_sequence_len,
        architecture="Embedding + BiLSTM + LSTM + Dense Softmax"
    )


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Predict candidate next words and probabilities given a prompt."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained or loaded yet. Run python training/train.py first."
        )
    try:
        response = predict_next_words(
            text=request.text,
            top_k=request.top_k,
            temperature=request.temperature
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/api/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: GenerateRequest):
    """Autoregressively generate next N words from a seed phrase."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained or loaded yet. Run python training/train.py first."
        )
    try:
        response = generate_sequence(
            seed_text=request.text,
            max_words=request.max_words,
            temperature=request.temperature,
            strategy=request.strategy
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation error: {str(e)}"
        )


# Static frontend mounting
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", tags=["Frontend"])
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
