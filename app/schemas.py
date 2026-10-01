"""
Pydantic schemas for Next Word Prediction API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input prompt text sequence", example="artificial intelligence is")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top candidate words to return")
    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description="Sampling temperature (lower = more confident)")


class CandidatePrediction(BaseModel):
    word: str
    probability: float
    confidence_percentage: str


class PredictResponse(BaseModel):
    input_text: str
    cleaned_input: str
    top_candidates: List[CandidatePrediction]
    best_prediction: str
    latency_ms: float


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Seed prompt", example="Natural language")
    max_words: int = Field(default=8, ge=1, le=50, description="Number of consecutive words to generate")
    temperature: float = Field(default=0.8, gt=0.0, le=2.0, description="Temperature scaling")
    strategy: str = Field(default="greedy", description="Generation strategy: 'greedy' or 'sample'")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    total_tokens_generated: int
    latency_ms: float


class ModelMetadata(BaseModel):
    model_name: str
    status: str
    vocab_size: int
    max_sequence_length: int
    architecture: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
