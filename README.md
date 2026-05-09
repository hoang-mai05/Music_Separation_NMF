# Supervised Non-negative Matrix Factorization for Audio Source Separation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and research for a two-stage, supervised Non-negative Matrix Factorization (NMF) pipeline designed to separate polyphonic music mixtures into isolated stems (drums, vocals, bass, and accompaniment). 

Unlike standard NMF architectures that apply uniform constraints across all sources, this project implements **decoupled, instrument-specific regularization**. By tailoring mathematical penalties to physical acoustic properties (e.g., sparsity for transient drums, temporal continuity for sustained bass), this pipeline mitigates penalty collapse and improves source isolation.

📄 **[Read the full research paper here](./paper/Music_Source_Separation.pdf)**

---

## Audio Examples

*(Note: Replace these links with the relative paths to your actual audio files)*
* [Original Mixture](./audio_samples/mixture.wav)
* [Isolated Drums (L1 Sparsity Penalized)](./audio_samples/drums_separated.wav)
* [Isolated Bass (Temporal Continuity Penalized)](./audio_samples/bass_separated.wav)

---

## Methodology

The traditional NMF objective function treats each time frame as an independent observation, ignoring the highly correlated nature of music. To address this, I implemented a three-stage pipeline:

### 1. Supervised Dictionary Training & Coherence Mitigation
I extracted isolated stems from the **MUSDB18 dataset** to train source-specific basis matrices ($B$). To mitigate basis coherence, I enforced explicit orthogonality by applying high-pass filtering to the vocal and harmonic dictionaries prior to separation.

### 2. Decoupled Regularization (The Optimization Stage)
During separation, the global dictionary is held constant while the activation matrix ($W$) is optimized using modified multiplicative update rules to minimize the Kullback-Leibler (KL) divergence. We appended instrument-specific penalties:
* **Drums (Transience):** An $L_1$ sparsity penalty to suppress the noise floor and enforce rapid acoustic decay.
* **Bass/Vocals (Sustain):** A variance-normalized Temporal Continuity (TC) penalty to enforce smooth, connected note envelopes without altering activation scale.

The modified update rule strictly separates positive and negative regularization gradients to ensure non-increasing convergence:

$$W_{new} = W_{old} \odot \frac{[\nabla_W C_{base}]^- + [\nabla_W J_{penalty}]^-}{[\nabla_W C_{base}]^+ + [\nabla_W J_{penalty}]^+}$$

### 3. Phase Recovery (Wiener Filtering)
Because NMF operates exclusively on magnitude spectrograms, raw reconstruction yields "robotic" phase artifacts. We utilized the NMF estimate to construct a soft Wiener mask, which was applied via Hadamard product to the original complex STFT, recovering the original phase prior to ISTFT conversion.

---

## Results & Insights

Performance was quantitatively evaluated using the Blind Source Separation (BSS) Evaluation toolkit (SDR, SIR, SAR). 

*(Note: Insert your `bss_evaluation_metrics.png` and `activation_plt.pgf/png` in the assets folder and link them here)*

**Key Findings:**
1. **The Sparsity Tradeoff:** Applying extreme $L_1$ penalties to drums successfully rejected harmonic bleed (increasing the Source-to-Interference Ratio, SIR), but truncated natural acoustic decay, resulting in a lower Source-to-Artifact Ratio (SAR).
2. **Interpretability over Black-Boxes:** While modern deep learning architectures (like Bi-LSTMs or U-Nets) achieve superior quantitative SDR, this decoupled NMF approach provides explicit, mathematically provable intuition linking matrix operations directly to physical acoustic traits.

---

## Execution Instructions

### Prerequisites
Download the MUSDB18 dataset

Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/NMF-Audio-Separation.git](https://github.com/yourusername/NMF-Audio-Separation.git)
cd NMF-Audio-Separation
pip install -r requirements.txt
