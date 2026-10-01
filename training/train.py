"""
Training script for the Next Word Prediction LSTM/GRU model.
"""

import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from training.preprocessing import load_corpus, prepare_sequences, save_tokenizer


def build_model(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 150,
    input_length: int = 20,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """Build a neural language model architecture for next-word prediction."""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=input_length,
            mask_zero=True,
            name="embedding_layer"
        ),
        Bidirectional(
            LSTM(lstm_units, return_sequences=True),
            name="bidirectional_lstm_1"
        ),
        Dropout(dropout_rate),
        LSTM(lstm_units, name="lstm_2"),
        Dropout(dropout_rate),
        Dense(lstm_units, activation="relu", name="dense_latent"),
        Dense(vocab_size, activation="softmax", name="output_logits")
    ], name="NextWordPredictor")

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        metrics=["accuracy"]
    )
    return model


def train(
    dataset_path: str = "data/dataset.txt",
    model_save_path: str = "model/next_word_model.keras",
    tokenizer_save_path: str = "model/tokenizer.pkl",
    config_save_path: str = "model/config.json",
    epochs: int = 60,
    batch_size: int = 32,
    embedding_dim: int = 128,
    lstm_units: int = 128
):
    """Execute training pipeline and save artifacts."""
    print("=== Next Word Prediction: Training Pipeline ===")
    print(f"Loading corpus from: {dataset_path}")
    corpus = load_corpus(dataset_path)

    print("Generating N-gram sequences and fitting tokenizer...")
    X, y, tokenizer, total_words, max_seq_len = prepare_sequences(corpus)
    input_len = max_seq_len - 1

    print(f"Total vocabulary size: {total_words}")
    print(f"Total training sequences: {len(X)}")
    print(f"Input sequence length: {input_len}")

    # Build model
    model = build_model(
        vocab_size=total_words,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        input_length=input_len
    )
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nStarting model training...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save tokenizer
    save_tokenizer(tokenizer, tokenizer_save_path)

    # Save metadata/config for inference consistency
    config = {
        "vocab_size": total_words,
        "max_sequence_len": max_seq_len,
        "input_len": input_len,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1])
    }
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration metadata saved to {config_save_path}")

    # Ensure model is saved in modern .keras format
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    print("=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next Word Prediction Model")
    parser.add_argument("--dataset", type=str, default="data/dataset.txt", help="Path to text dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-out", type=str, default="model/next_word_model.keras", help="Model output path")
    parser.add_argument("--tokenizer-out", type=str, default="model/tokenizer.pkl", help="Tokenizer output path")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        model_save_path=args.model_out,
        tokenizer_save_path=args.tokenizer_out,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
