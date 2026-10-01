"""
Preprocessing utilities for inference pipeline.
"""

import re
from typing import List, Optional
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_input_text(text: str) -> str:
    """Sanitize user input text for inference."""
    text = text.lower()
    # Normalize punctuation and unwanted characters
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def encode_prompt(
    text: str,
    tokenizer: Tokenizer,
    max_len: int
) -> np.ndarray:
    """
    Convert raw prompt to padded sequence array for model input.
    """
    cleaned = clean_input_text(text)
    if not cleaned:
        return np.zeros((1, max_len), dtype=np.int32)
        
    sequence = tokenizer.texts_to_sequences([cleaned])[0]
    padded = pad_sequences([sequence], maxlen=max_len, padding="pre")
    return padded
