"""
Evaluation script to compute Top-k accuracy, perplexity, and qualitative completions.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from training.preprocessing import load_corpus, prepare_sequences, load_tokenizer


def calculate_perplexity(loss: float) -> float:
    """Perplexity = exp(cross_entropy_loss)."""
    return float(np.exp(loss))


def evaluate_model(
    dataset_path: str = "data/dataset.txt",
    model_path: str = "model/next_word_model.keras",
    tokenizer_path: str = "model/tokenizer.pkl",
    config_path: str = "model/config.json"
):
    print("=== Next Word Prediction: Model Evaluation ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}.")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    corpus = load_corpus(dataset_path)
    X, y, _, total_words, max_seq_len = prepare_sequences(corpus, tokenizer=tokenizer)

    print(f"Evaluating across {len(X)} test n-gram pairs...")
    evaluation = model.evaluate(X, y, verbose=0)
    loss, accuracy = evaluation[0], evaluation[1]
    perplexity = calculate_perplexity(loss)

    # Compute Top-3 and Top-5 accuracy
    predictions = model.predict(X, verbose=0)
    top_3_acc = np.mean([y[i] in np.argsort(predictions[i])[-3:] for i in range(len(y))])
    top_5_acc = np.mean([y[i] in np.argsort(predictions[i])[-5:] for i in range(len(y))])

    metrics = {
        "loss": float(loss),
        "top_1_accuracy": float(accuracy),
        "top_3_accuracy": float(top_3_acc),
        "top_5_accuracy": float(top_5_acc),
        "perplexity": float(perplexity),
        "test_samples": len(X)
    }

    print("\n" + "=" * 40)
    print("           EVALUATION METRICS          ")
    print("=" * 40)
    print(f"Loss (Cross-Entropy): {metrics['loss']:.4f}")
    print(f"Perplexity:           {metrics['perplexity']:.4f}")
    print(f"Top-1 Accuracy:       {metrics['top_1_accuracy'] * 100:.2f}%")
    print(f"Top-3 Accuracy:       {metrics['top_3_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy:       {metrics['top_5_accuracy'] * 100:.2f}%")
    print("=" * 40)

    # Qualitative sample test
    sample_prompts = [
        "artificial intelligence is",
        "deep learning models",
        "natural language",
        "the quick brown",
        "machine learning engineers train"
    ]

    print("\nQualitative Completion Tests:")
    input_len = max_seq_len - 1
    for prompt in sample_prompts:
        seq = tokenizer.texts_to_sequences([prompt.lower()])[0]
        padded = pad_sequences([seq], maxlen=input_len, padding="pre")
        preds = model.predict(padded, verbose=0)[0]
        top_indices = np.argsort(preds)[-3:][::-1]
        
        idx_to_word = {v: k for k, v in tokenizer.word_index.items()}
        top_words = [f"{idx_to_word.get(idx, '<unk>')} ({preds[idx]*100:.1f}%)" for idx in top_indices]
        print(f"Prompt: \"{prompt}\" -> Predicted: {', '.join(top_words)}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Next Word Prediction Model")
    parser.add_argument("--dataset", type=str, default="data/dataset.txt")
    parser.add_argument("--model", type=str, default="model/next_word_model.keras")
    parser.add_argument("--tokenizer", type=str, default="model/tokenizer.pkl")
    args = parser.parse_args()

    evaluate_model(
        dataset_path=args.dataset,
        model_path=args.model,
        tokenizer_path=args.tokenizer
    )
