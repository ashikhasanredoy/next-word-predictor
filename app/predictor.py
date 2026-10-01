"""
Prediction and text generation engine.
"""

import time
import numpy as np
from typing import List, Tuple
from app.model import model_manager
from app.preprocessing import clean_input_text, encode_prompt
from app.schemas import CandidatePrediction, PredictResponse, GenerateResponse


def apply_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to raw probabilities or logits."""
    temperature = max(temperature, 1e-4)
    # Log transform probabilities for safe softmax temperature scaling
    log_probs = np.log(np.clip(logits, 1e-10, 1.0)) / temperature
    exp_probs = np.exp(log_probs - np.max(log_probs))
    return exp_probs / np.sum(exp_probs)


def predict_next_words(
    text: str,
    top_k: int = 5,
    temperature: float = 1.0
) -> PredictResponse:
    """Predict the top-k next words for a given text prompt."""
    start_time = time.time()
    cleaned = clean_input_text(text)

    if not model_manager.is_loaded:
        raise RuntimeError("Model is not loaded. Please train the model using 'python -m training.train' first.")

    model = model_manager.model
    tokenizer = model_manager.tokenizer
    input_len = model_manager.max_sequence_len

    # Encode input
    input_tensor = encode_prompt(cleaned, tokenizer, max_len=input_len)

    # Predict distribution
    raw_preds = model.predict(input_tensor, verbose=0)[0]
    scaled_probs = apply_temperature(raw_preds, temperature)

    # Get top-k indices (ignoring 0 index / padding and OOV if needed)
    top_indices = np.argsort(scaled_probs)[::-1]
    
    candidates: List[CandidatePrediction] = []
    idx_map = model_manager.index_to_word

    for idx in top_indices:
        if idx == 0:
            continue
        word = idx_map.get(idx)
        if word and word != "<OOV>":
            prob = float(scaled_probs[idx])
            candidates.append(CandidatePrediction(
                word=word,
                probability=round(prob, 4),
                confidence_percentage=f"{prob * 100:.1f}%"
            ))
            if len(candidates) >= top_k:
                break

    best_word = candidates[0].word if candidates else ""
    latency = (time.time() - start_time) * 1000

    return PredictResponse(
        input_text=text,
        cleaned_input=cleaned,
        top_candidates=candidates,
        best_prediction=best_word,
        latency_ms=round(latency, 2)
    )


def generate_sequence(
    seed_text: str,
    max_words: int = 8,
    temperature: float = 0.8,
    strategy: str = "greedy"
) -> GenerateResponse:
    """Autoregressively generate next N words from a seed phrase."""
    start_time = time.time()
    current_text = seed_text
    tokens_generated = 0

    if not model_manager.is_loaded:
        raise RuntimeError("Model is not loaded. Train the model first.")

    for _ in range(max_words):
        pred_res = predict_next_words(current_text, top_k=5, temperature=temperature)
        if not pred_res.top_candidates:
            break

        if strategy == "sample" and len(pred_res.top_candidates) > 1:
            words = [c.word for c in pred_res.top_candidates]
            probs = [c.probability for c in pred_res.top_candidates]
            probs = np.array(probs) / np.sum(probs)
            chosen_word = np.random.choice(words, p=probs)
        else:
            chosen_word = pred_res.best_prediction

        if not chosen_word:
            break

        current_text += " " + chosen_word
        tokens_generated += 1

    latency = (time.time() - start_time) * 1000

    return GenerateResponse(
        prompt=seed_text,
        generated_text=current_text,
        total_tokens_generated=tokens_generated,
        latency_ms=round(latency, 2)
    )
