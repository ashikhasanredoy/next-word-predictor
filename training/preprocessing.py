"""
Data Preprocessing module for training the Next Word Prediction model.
"""

import os
import re
import pickle
from typing import Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text: str) -> str:
    """Clean raw text corpus."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_corpus(file_path: str) -> str:
    """Load and clean dataset text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return "\n".join(lines)


def prepare_sequences(
    corpus_text: str,
    tokenizer: Tokenizer = None,
    max_sequence_len: int = None
) -> Tuple[np.ndarray, np.ndarray, Tokenizer, int, int]:
    """
    Generate n-gram sequences from text and prepare training features (X) and labels (y).
    
    Returns:
        X: padded input sequences (N, max_sequence_len - 1)
        y: target word indices (N,)
        tokenizer: fitted Tokenizer
        total_words: vocabulary size + 1 (for 0-padding)
        max_sequence_len: length of longest n-gram sequence
    """
    lines = [clean_text(line) for line in corpus_text.split("\n") if line.strip()]
    
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(lines)
        
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences: List[List[int]] = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
            
    if not input_sequences:
        raise ValueError("No valid sequences generated from corpus.")
        
    if max_sequence_len is None:
        max_sequence_len = max(len(seq) for seq in input_sequences)
        
    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )
    
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    
    return X, y, tokenizer, total_words, max_sequence_len


def save_tokenizer(tokenizer: Tokenizer, save_path: str) -> None:
    """Serialize and save tokenizer to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved successfully to {save_path}")


def load_tokenizer(load_path: str) -> Tokenizer:
    """Load serialized tokenizer from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Tokenizer not found at {load_path}")
    with open(load_path, "rb") as f:
        return pickle.load(f)
