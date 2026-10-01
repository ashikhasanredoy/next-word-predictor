/**
 * Next Word Prediction UI Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  const userInput = document.getElementById("user-input");
  const btnGenerate = document.getElementById("btn-generate");
  const btnClear = document.getElementById("btn-clear");
  const candidateChips = document.getElementById("candidate-chips");
  const probabilitiesContainer = document.getElementById("probabilities-container");
  const latencyBadge = document.getElementById("latency-badge");

  const topKSlider = document.getElementById("top-k-slider");
  const topKVal = document.getElementById("top-k-val");
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const genLenSlider = document.getElementById("gen-len-slider");
  const genLenVal = document.getElementById("gen-len-val");

  let debounceTimer = null;
  let currentTopWord = null;

  // Sliders
  topKSlider.addEventListener("input", (e) => {
    topKVal.textContent = e.target.value;
    triggerPrediction();
  });

  tempSlider.addEventListener("input", (e) => {
    tempVal.textContent = parseFloat(e.target.value).toFixed(1);
    triggerPrediction();
  });

  genLenSlider.addEventListener("input", (e) => {
    genLenVal.textContent = e.target.value;
  });

  // Check health
  async function checkHealth() {
    try {
      await fetch("/health");
    } catch {
      // Backend status
    }
  }

  // Debounced Predict Trigger
  function triggerPrediction() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fetchPredictions, 200);
  }

  userInput.addEventListener("input", triggerPrediction);

  // Tab key auto-complete
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && currentTopWord) {
      e.preventDefault();
      appendWord(currentTopWord);
    }
  });

  // Debounced Predict Trigger

  // Action buttons
  btnGenerate.addEventListener("click", fetchGeneration);
  btnClear.addEventListener("click", () => {
    userInput.value = "";
    currentTopWord = null;
    candidateChips.innerHTML = '<span class="empty-hint">Type above to see predictions in real time</span>';
    probabilitiesContainer.innerHTML = '<div class="empty-prob"><p>Probabilities appear as you type.</p></div>';
    latencyBadge.classList.add("hidden");
    userInput.focus();
  });

  // Append word to input
  function appendWord(word) {
    const text = userInput.value.trim();
    userInput.value = text.length > 0 ? `${text} ${word}` : word;
    userInput.focus();
    fetchPredictions();
  }

  // Predict
  async function fetchPredictions() {
    const text = userInput.value.trim();
    if (!text) {
      currentTopWord = null;
      candidateChips.innerHTML = '<span class="empty-hint">Type above to see predictions in real time</span>';
      probabilitiesContainer.innerHTML = '<div class="empty-prob"><p>Probabilities appear as you type.</p></div>';
      latencyBadge.classList.add("hidden");
      return;
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          top_k: parseInt(topKSlider.value, 10),
          temperature: parseFloat(tempSlider.value)
        })
      });

      if (!res.ok) return;

      const data = await res.json();
      renderPredictions(data);
    } catch (e) {
      console.error(e);
    }
  }

  // Generate sequence
  async function fetchGeneration() {
    const text = userInput.value.trim();
    if (!text) return;

    btnGenerate.disabled = true;
    btnGenerate.classList.add("loading");

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          max_words: parseInt(genLenSlider.value, 10),
          temperature: parseFloat(tempSlider.value),
          strategy: "sample"
        })
      });

      if (res.ok) {
        const data = await res.json();
        userInput.value = data.generated_text;
        fetchPredictions();
      }
    } catch (e) {
      console.error(e);
    } finally {
      btnGenerate.disabled = false;
      btnGenerate.classList.remove("loading");
    }
  }

  // Render UI
  function renderPredictions(data) {
    latencyBadge.textContent = `${data.latency_ms} ms`;
    latencyBadge.classList.remove("hidden");

    const candidates = data.top_candidates || [];
    if (candidates.length === 0) {
      candidateChips.innerHTML = '<span class="empty-hint">No matches found</span>';
      probabilitiesContainer.innerHTML = '<div class="empty-prob"><p>No probabilities available.</p></div>';
      currentTopWord = null;
      return;
    }

    currentTopWord = candidates[0].word;

    // Render chips
    candidateChips.innerHTML = "";
    candidates.forEach((cand, idx) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `word-chip ${idx === 0 ? "best-match" : ""}`;
      chip.innerHTML = `
        <span class="chip-text">${cand.word}</span>
        <span class="chip-confidence">${cand.confidence_percentage}</span>
      `;
      chip.addEventListener("click", () => appendWord(cand.word));
      candidateChips.appendChild(chip);
    });

    // Render probability bars
    probabilitiesContainer.innerHTML = "";
    candidates.forEach((cand) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      const pct = Math.min(100, Math.max(0, cand.probability * 100));

      row.innerHTML = `
        <div class="prob-labels">
          <span class="prob-word">${cand.word}</span>
          <span class="prob-pct">${cand.confidence_percentage}</span>
        </div>
        <div class="prob-track">
          <div class="prob-bar" style="width: ${pct}%"></div>
        </div>
      `;
      probabilitiesContainer.appendChild(row);
    });
  }

  checkHealth();
});
