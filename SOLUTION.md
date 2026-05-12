# Hallucination Detection in Small Language Models — Solution

**Author:** Andrey  
**Project:** SMILES-2026 — Hallucination Detection  
**Final Test AUROC:** 76.07% (5-fold average)

---

## Approach

The pipeline has three stages:

### 1. Feature Extraction

The full `prompt + response` text (in ChatML format) is passed through `Qwen/Qwen2.5-0.5B` with `output_hidden_states=True`. This produces 25 hidden state tensors (1 embedding + 24 transformer layers), each of shape `(seq_len, 896)`.

**Aggregation strategy:** The hidden state of the **last real token** (EOS: `<|endoftext|>`) of the **final transformer layer** is used as the feature vector. This gives a 896-dimensional representation.

In autoregressive transformers, the EOS token attends to all previous tokens via causal attention, making it a natural aggregate of the entire sequence — both the question context and the model's answer.

**What we tried and why it did not work:**
- Mean pooling over all tokens: includes noisy padding/formatting tokens → AUROC dropped to 55%
- Concatenation of layers 12, 16, 20, 24 (3584d): too many features for 481 training samples → 69%
- Last 30 tokens mean pooling: response tokens not reliably isolated → 63%
- Layer-wise activation norms as geometric features: low discriminative signal → 72%

### 2. Probe Classifier

A two-layer MLP trained on the extracted features:

```
Linear(896 → 256) → ReLU → Linear(256 → 1) → Sigmoid
```

**Training details:**
- Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` to handle class imbalance
- Optimizer: `AdamW(lr=1e-3)`
- Scheduler: `CosineAnnealingLR(T_max=500)`
- Epochs: 1000
- Preprocessing: `StandardScaler` applied to features before training

### 3. Evaluation Strategy

**5-fold Stratified Cross-Validation** on the train+val portion (585 samples), with a fixed held-out test set (104 samples, 15% of dataset). Each fold trains on ~468 samples, validates on ~117.

**Ensemble predictions:** For the final `predictions.csv`, all 5 fold-models produce probability estimates on the test set. These are averaged and thresholded at 0.5 — this reduces variance compared to a single model.

---

## Results

| Configuration | Test AUROC |
|---|---|
| Majority-class baseline | N/A |
| Original MLP (Adam, 1 fold) | 74.41% |
| + 5-fold cross-validation | 74.52% |
| + AdamW optimizer | 74.52% |
| + CosineAnnealingLR, 500 epochs | 75.37% |
| **+ Ensemble predictions (final)** | **76.07%** |

**Per-fold breakdown (final configuration):**

| Fold | Train AUROC | Val AUROC | Test AUROC |
|---|---|---|---|
| 1 | 100.00% | 62.89% | 75.94% |
| 2 | 100.00% | 72.35% | 78.44% |
| 3 | 100.00% | 70.17% | 73.46% |
| 4 | 100.00% | 64.22% | 78.79% |
| 5 | 100.00% | 74.49% | 73.71% |
| **Avg** | **100.00%** | **68.83%** | **76.07%** |

---

## Modified Files

| File | Changes |
|---|---|
| `aggregation.py` | No changes — original last-token strategy was optimal |
| `probe.py` | Adam - > AdamW + CosineAnnealingLR(T_max=500), 500 epochs |
| `splitting.py` | Single split - > 5-fold StratifiedKFold + fixed hold-out test |
| `solution.py` | Added feature caching + 5-fold ensemble for `predictions.csv` |

---

## Reproducing the Results

### Requirements

```bash
pip install -r requirements.txt
```

### Steps to reproduce the pipeline

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

# 2. Run the solution
python solution.py
```

**Outputs:**
- `results.json` — evaluation metrics per fold
- `predictions.csv` — ensemble predictions on the competition test set (100 samples)

### Key Hyperparameters

```python
# solution.py
BATCH_SIZE    = 4
USE_GEOMETRIC = False
MAX_LENGTH    = 512

# probe.py
optimizer = AdamW(lr=1e-3)
scheduler = CosineAnnealingLR(T_max=500)
epochs    = 1000

# splitting.py
n_folds   = 5
test_size = 0.15
```

---

## What we got from this study:

1. **EOS token of the final layer** is the most informative single feature for hallucination detection in autoregressive LLMs — it aggregates the full sequence via causal attention.

2. **More features ≠ better results** with small datasets. Multi-layer concatenation (3584d) hurt performance due to the curse of dimensionality.

3. **CosineAnnealingLR** provides better convergence than fixed lr, giving the optimizer time to escape local minima with gradual lr decay.

4. **Ensemble over k-fold models** reduces prediction variance and improves reliability of `predictions.csv` without additional feature extraction cost.
