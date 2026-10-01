"""
Model and Tokenizer Loader with Singleton pattern and fallback support.
"""

import os
import json
import pickle
import numpy as np
from typing import Optional, Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


class ModelManager:
    _instance = None
    _model: Optional[tf.keras.Model] = None
    _tokenizer: Optional[Tokenizer] = None
    _config: Dict[str, Any] = {}
    _index_to_word: Dict[int, str] = {}
    _is_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def load_resources(
        self,
        model_path: str = "model/next_word_model.keras",
        tokenizer_path: str = "model/tokenizer.pkl",
        config_path: str = "model/config.json"
    ) -> bool:
        """Load trained neural network model and tokenizer from disk."""
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)

            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, "rb") as f:
                    self._tokenizer = pickle.load(f)
                self._index_to_word = {idx: word for word, idx in self._tokenizer.word_index.items()}

            if os.path.exists(model_path):
                self._model = tf.keras.models.load_model(model_path)
                self._is_loaded = True
                print(f"[ModelManager] Model & Tokenizer loaded successfully from {model_path}")
                return True
            else:
                print(f"[ModelManager] Model file {model_path} not found. Please run training/train.py.")
                self._is_loaded = False
                return False
        except Exception as e:
            print(f"[ModelManager] Error loading model resources: {e}")
            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None and self._tokenizer is not None

    @property
    def model(self) -> Optional[tf.keras.Model]:
        return self._model

    @property
    def tokenizer(self) -> Optional[Tokenizer]:
        return self._tokenizer

    @property
    def index_to_word(self) -> Dict[int, str]:
        return self._index_to_word

    @property
    def max_sequence_len(self) -> int:
        return self._config.get("input_len", 20)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer:
            return len(self._tokenizer.word_index) + 1
        return self._config.get("vocab_size", 0)


model_manager = ModelManager()
