# Next Word Prediction — Deep Learning Language Model

A production-ready **Next Word Prediction** system built with a **Bidirectional LSTM neural language model**, served via a **FastAPI** backend and a clean **glassmorphism web interface**. The model learns statistical patterns from text data and predicts the most likely next word(s) for any given input sequence.

---

## 📐 How It Works — Architecture Overview

```
Raw Text Corpus
      ↓
  Text Cleaning (lowercase, remove special chars)
      ↓
  Tokenizer (word → unique integer index)
      ↓
  N-gram Sequence Generation
      ↓
  Pre-Padding (uniform sequence length)
      ↓
  Feature Split: X (input context) / y (target next word)
      ↓
  Embedding Layer (integer → dense vector)
      ↓
  Bidirectional LSTM (forward + backward context)
      ↓
  LSTM Layer (sequence-to-vector)
      ↓
  Dense + Dropout (regularization)
      ↓
  Softmax Output Layer (probability distribution over vocabulary)
      ↓
  Top-K Prediction / Temperature Sampling
```

### Detailed Stage-by-Stage Explanation

| Stage | Description |
|---|---|
| **Text Cleaning** | Converts corpus to lowercase. Strips punctuation and special characters. Normalizes whitespace. |
| **Tokenizer** | Assigns each unique word a unique integer ID. `"artificial" → 1`, `"intelligence" → 2`. Vocabulary is built from the full corpus. |
| **N-gram Generation** | Every sentence is turned into overlapping subsequences. `"deep learning models"` becomes `[deep, learning]` → `models`, `[deep]` → `learning`, etc. |
| **Pre-Padding** | All sequences are padded with zeros at the front to reach `max_sequence_len`. Ensures uniform tensor shapes. |
| **Feature Split** | Each padded sequence is split: all tokens except last = input `X`; the last token = label `y`. |
| **Embedding Layer** | Maps each word index to a 128-dimensional dense vector that encodes semantic similarity. Trained jointly with the model. |
| **Bidirectional LSTM** | Processes sequences left-to-right AND right-to-left simultaneously, capturing richer contextual patterns. |
| **LSTM Layer** | Compresses the sequential output into a fixed-size vector representation. |
| **Dropout** | Randomly zeroes 20% of neurons during training to prevent memorization and improve generalization. |
| **Softmax Output** | Produces a probability distribution over every word in the vocabulary. The highest probability index = the predicted next word. |

---

## 📁 Project Structure

```
next-word-prediction/
│
├── app/                          # FastAPI backend application
│   ├── __init__.py               # Package declaration
│   ├── main.py                   # FastAPI server, API routes, static file serving
│   ├── model.py                  # Singleton ModelManager: loads/caches model & tokenizer
│   ├── predictor.py              # Core prediction engine & autoregressive text generator
│   ├── preprocessing.py          # Inference-time text cleaning & sequence encoding
│   └── schemas.py                # Pydantic request & response validation schemas
│
├── model/                        # Generated artifacts after training
│   ├── next_word_model.keras     # Trained Keras model weights & architecture
│   ├── tokenizer.pkl             # Pickled word-to-index tokenizer vocabulary
│   └── config.json               # Training metadata (vocab size, sequence length, etc.)
│
├── training/                     # Offline training pipeline
│   ├── train.py                  # Full training script: build, compile, fit, save model
│   ├── preprocessing.py          # Corpus loading, N-gram generation, padding
│   └── evaluate.py               # Metrics: Top-k accuracy, perplexity, qualitative tests
│
├── data/
│   └── dataset.txt               # Text corpus (732+ lines across 12 topic domains)
│
├── frontend/                     # Web user interface
│   ├── index.html                # Application layout and HTML structure
│   ├── style.css                 # Glassmorphism dark theme, animations, typography
│   └── script.js                 # Real-time prediction, Tab autocomplete, sliders
│
├── requirements.txt              # Python dependency list
├── README.md                     # This file
└── .gitignore                    # Ignored files and directories
```

---

## 🗂️ Dataset Topics

The training corpus (`data/dataset.txt`) covers **12 distinct domains** to give the model broad predictive coverage:

| # | Domain | Example Topics |
|---|---|---|
| 1 | **Artificial Intelligence & ML** | Neural networks, LSTM, embeddings, transformers |
| 2 | **Python Programming** | Libraries, syntax, data science, automation |
| 3 | **Data Science** | Feature engineering, EDA, model evaluation |
| 4 | **Computer Vision** | CNNs, image classification, object detection |
| 5 | **NLP & Language Models** | Tokenization, text generation, embeddings |
| 6 | **Astronomy & Space** | Solar system, black holes, space telescopes |
| 7 | **Biology & Medicine** | DNA, immune system, cellular respiration |
| 8 | **Climate & Environment** | Renewable energy, deforestation, conservation |
| 9 | **Economics & Finance** | Supply/demand, fiscal policy, investment |
| 10 | **History & Philosophy** | Ancient civilizations, ethics, critical thinking |
| 11 | **Cybersecurity & Cloud** | Encryption, firewalls, microservices, DevOps |
| 12 | **Daily Life & Habits** | Communication, productivity, health, travel |

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Virtual Environment

```bash
# Create a Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥2.15 | Keras model training and inference |
| `fastapi` | ≥0.110 | REST API server framework |
| `uvicorn` | ≥0.28 | ASGI web server |
| `pydantic` | ≥2.6 | Request/response data validation |
| `numpy` | ≥1.24 | Numerical operations |
| `scikit-learn` | ≥1.4 | Supplementary ML utilities |

---

### 3. Train the Model

```bash
python -m training.train --epochs 50 --batch-size 32
```

**Optional Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `50` | Number of full passes through the dataset |
| `--batch-size` | `32` | Number of sequences per gradient update |
| `--dataset` | `data/dataset.txt` | Path to training corpus |
| `--model-out` | `model/next_word_model.keras` | Where to save the trained model |
| `--tokenizer-out` | `model/tokenizer.pkl` | Where to save the tokenizer |

**What happens during training:**
1. The corpus is loaded and cleaned.
2. N-gram sequences are generated for every sentence.
3. Sequences are padded and split into `(X, y)` pairs.
4. The Bidirectional LSTM model is compiled and trained.
5. `EarlyStopping` halts training if loss stops improving (patience=10).
6. `ReduceLROnPlateau` halves the learning rate on stagnation.
7. `ModelCheckpoint` saves only the best model to disk.
8. The tokenizer and config metadata are saved alongside the model.

---

### 4. Evaluate the Model

```bash
python -m training.evaluate
```

**Metrics computed:**

| Metric | Description |
|---|---|
| **Loss (Cross-Entropy)** | Average negative log-likelihood over all N-gram pairs |
| **Perplexity** | `exp(loss)` — lower is better; measures prediction uncertainty |
| **Top-1 Accuracy** | % of times the model's #1 prediction matches the true next word |
| **Top-3 Accuracy** | % of times the true word appears in the top 3 predictions |
| **Top-5 Accuracy** | % of times the true word appears in the top 5 predictions |

**Latest Results (732-line multi-topic dataset):**
```
Loss (Cross-Entropy): 0.6505
Perplexity:           1.9164
Top-1 Accuracy:       81.52%
Top-3 Accuracy:       92.34%
Top-5 Accuracy:       94.85%
```

---

### 5. Launch the Web Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

| URL | Description |
|---|---|
| `http://localhost:8080` | Interactive glassmorphism web interface |
| `http://localhost:8080/docs` | Auto-generated Swagger API documentation |
| `http://localhost:8080/health` | Server and model health status |

---

## 📡 REST API Reference

### `GET /health`
Returns the server and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /api/predict`
Returns the top-K predicted next words for a given text prompt.

**Request Body:**
```json
{
  "text": "Renewable energy sources",
  "top_k": 5,
  "temperature": 1.0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Input prompt sentence |
| `top_k` | `int` | `5` | Number of candidate words to return (1–20) |
| `temperature` | `float` | `1.0` | Sampling temperature: lower = more confident, higher = more creative |

**Response:**
```json
{
  "input_text": "Renewable energy sources",
  "cleaned_input": "renewable energy sources",
  "top_candidates": [
    { "word": "such", "probability": 0.8421, "confidence_percentage": "84.2%" },
    { "word": "can",  "probability": 0.0912, "confidence_percentage": "9.1%"  }
  ],
  "best_prediction": "such",
  "latency_ms": 62.4
}
```

---

### `POST /api/generate`
Autoregressively generates the next N words from a seed phrase.

**Request Body:**
```json
{
  "text": "Cybersecurity protects",
  "max_words": 6,
  "temperature": 0.8,
  "strategy": "greedy"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Seed phrase |
| `max_words` | `int` | `8` | How many words to generate (1–50) |
| `temperature` | `float` | `0.8` | Sampling creativity (0.1–2.0) |
| `strategy` | `str` | `"greedy"` | `"greedy"` (deterministic) or `"sample"` (stochastic) |

**Response:**
```json
{
  "prompt": "Cybersecurity protects",
  "generated_text": "Cybersecurity protects computer networks and digital data from",
  "total_tokens_generated": 6,
  "latency_ms": 387.5
}
```

---

### `GET /api/metadata`
Returns model architecture and vocabulary details.

**Response:**
```json
{
  "model_name": "LSTM Next Word Predictor",
  "status": "active",
  "vocab_size": 1024,
  "max_sequence_length": 28,
  "architecture": "Embedding + BiLSTM + LSTM + Dense Softmax"
}
```

---

## 🖥️ Web Interface Features

| Feature | Description |
|---|---|
| **Live Prediction** | Predictions update in real time as you type (200ms debounce) |
| **Tab Autocomplete** | Press `Tab` to instantly append the top predicted word |
| **Click-to-Accept** | Click any suggestion chip to append it to the text |
| **Continue Generating** | Auto-generates a sequence of next words from your current prompt |
| **Top-K Slider** | Control how many candidate words appear (1–10) |
| **Temperature Slider** | Adjust prediction creativity (0.1 = focused, 2.0 = random) |
| **Generation Length** | Set how many words to generate (2–20) |
| **Probability Bars** | Visual softmax probability distribution for each candidate |
| **Latency Display** | Shows inference time in milliseconds per prediction |

---

## 🧠 Model Architecture Details

```
Model: NextWordPredictor
────────────────────────────────────────────────
Layer                    Output Shape     Params
────────────────────────────────────────────────
Embedding (128d)         (None, len, 128)  vocab × 128
Bidirectional LSTM (128) (None, len, 256)  ~264K
Dropout (0.2)            (None, len, 256)  0
LSTM (128)               (None, 128)       ~197K
Dropout (0.2)            (None, 128)       0
Dense ReLU (128)         (None, 128)       ~16K
Dense Softmax (vocab)    (None, vocab)     ~131K
────────────────────────────────────────────────
Optimizer: Adam (lr=0.003)
Loss: Sparse Categorical Cross-Entropy
```

---

## ⚙️ Key Design Decisions

**Why Bidirectional LSTM?**
A standard LSTM reads sequences only left-to-right. A Bidirectional LSTM reads in both directions simultaneously, giving each token richer contextual information from both past and future tokens in the sequence.

**Why Temperature Scaling?**
At `temperature = 1.0`, the model uses the raw softmax probabilities. Below 1.0, predictions become sharper (more deterministic). Above 1.0, the distribution flattens, introducing more diversity and creativity in generation.

**Why Pre-Padding?**
Padding is added at the beginning (pre-padding) rather than the end. For next-word prediction, the most recent tokens are the most relevant — placing them closest to the output end of the sequence helps the LSTM retain the most useful context.

**Why Singleton ModelManager?**
Loading a Keras model is expensive. The `ModelManager` loads the model once at startup and caches it in memory. All incoming prediction requests share the same model instance, keeping latency low.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow / Keras 2.16 |
| **Model Architecture** | Bidirectional LSTM + Embedding |
| **API Framework** | FastAPI + Uvicorn |
| **Data Validation** | Pydantic v2 |
| **Frontend** | Vanilla HTML5, CSS3 (Glassmorphism), ES6 JavaScript |
| **Typography** | Plus Jakarta Sans + JetBrains Mono (Google Fonts) |
| **Serialization** | `.keras` format (model), `pickle` (tokenizer) |
