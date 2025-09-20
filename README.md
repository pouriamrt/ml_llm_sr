
# MASLD RCT Paper Screener (LLM + Supervised + Fusion)

End‑to‑end pipeline to screen ~6k research papers using **Title + Abstract** only, label them as **include / exclude / maybe**, and fuse:
1) a calibrated **LightGBM** classifier trained on your gold labels,  
2) an **LLM rater** (OpenAI) guided by strict domain prompts, and  
3) a small **keyword heuristic** aligned to the study scope.

The project targets **adult MASLD/NAFLD** randomized controlled trials (RCTs) whose interventions aim to reduce hepatic fat, with comparator groups and liver‑related outcomes. It outputs audit‑friendly predictions with probabilities, LLM reasons, and verbatim evidence spans.

---

## Why this repo

- **Domain‑tight prompts** encode the inclusion/exclusion criteria.
- **Async LLM labeling** for speed (rate‑limit aware).
- **Calibrated LightGBM** on sentence‑embeddings for robust probabilities.
- **Meta‑fusion** (logistic regression) combines supervised probs, LLM probs, and keyword heuristic.
- **Optional self‑training** to iteratively improve the supervised model using confident pseudo‑labels.
- **Reproducible config** via `config.py`.

---

## Repository Layout

```
.gitignore
.python-version          # 3.13
pyproject.toml           # dependency lock
config.py                # paths + hyperparams
prompts.py               # system/user prompts + fewshots
utils.py                 # helpers (prep text, heuristics, io)
data_prep.py             # build text field + embeddings
train_supervised.py      # LightGBM + isotonic calibration
llm_label.py             # async OpenAI labeling → parquet
fuse_and_predict.py      # meta-fusion (supervised + LLM + kw)
self_training.py         # optional pseudo-label retrain
run_pipeline.py          # orchestration script
main.py                  # placeholder
```

---

## Data expectations

Input CSVs under `data/` (configurable in `config.py → Paths`):

- **`labeled_include.csv`**, **`labeled_exclude.csv`**, optional **`labeled_maybe.csv`**
- **`unlabeled_6000.csv`**

Required columns (order not important):

| Column          | Description                                               |
|-----------------|-----------------------------------------------------------|
| `Key`           | Stable paper identifier (string)                          |
| `Title`         | Paper title                                               |
| `Abstract Note` | Abstract text                                             |
| `Manual Tags`   | Optional keywords / tags (semicolon-separated)            |

During `data_prep.py` these are concatenated into a single **`text`** field.

---

## Installation

> Requires **Python 3.13** (see `.python-version`).

```bash
# create and activate a virtual env (choose any tool you like)
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
# OR: source .venv/bin/activate (macOS/Linux)

# install deps
pip install -U pip
pip install -e .
# or: pip install -r <generated requirements> if you prefer
```

Set your OpenAI API key for the LLM step:

```bash
# .env (in project root)
OPENAI_API_KEY=sk-...
```

> You can also export it in your shell; `LLMCfg.api_key` reads from env.

---

## Quickstart (one‑shot)

```bash
python run_pipeline.py
```

This executes:

1. **`data_prep.py`** → builds `artifacts/embeddings/{gold,unl}.parquet` + `.npy`
2. **`train_supervised.py`** → trains calibrated LightGBM → `artifacts/supervised/supervised_clf.pkl`
3. **`llm_label.py`** → async OpenAI labeling → `artifacts/llm/llm_labels_{gold,unl}.parquet`
4. **`fuse_and_predict.py`** → meta‑fusion & final predictions → `outputs/final_predictions.parquet`
5. (Optional) **`self_training.py`** → retrains supervised model with confident pseudo‑labels

You can run scripts individually to inspect each stage.

---

## Configuration (`config.py`)

### Paths
Edit `Paths` to point to your CSVs and artifact folders.

### ModelCfg (supervised + fusion)
- `embed_model`: sentence‑transformers model (default: `all-mpnet-base-v2`)
- LightGBM knobs: `lightgbm_*` (depth, leaves, LR, etc.)
- `maybe_low`, `maybe_high`: probability band for **maybe** (default 0.40–0.60)
- `include_weight`: increases importance of the **include** class in fusion
- `pseudo_min_prob`: confidence threshold for pseudo‑labels in self‑training

### LLMCfg
- `model`: OpenAI chat model (default: `gpt-4.1-mini`)
- `temperature`, `max_retries`, `concurrency`, `timeout_s`
- `confidence_map`: maps LLM `"high"/"medium"/"low"` to a numeric include‑prob proxy
- `use_fewshots`: include examples from `prompts.py`

---

## How it works

### 1) Embeddings (`data_prep.py`)
- Loads labeled + unlabeled CSVs, builds `text = Title + Abstract + Manual Tags`.
- Encodes with `SentenceTransformer` into `gold.npy` / `unl.npy`.
- Saves `gold.parquet` / `unl.parquet` with the raw text for downstream steps.

### 2) Supervised model (`train_supervised.py`)
- Trains **LightGBM** (binary or multiclass depending on labels present).
- Wraps with **Isotonic Calibration** via `CalibratedClassifierCV` for good `predict_proba`.
- If binary, converts calibrated `P(include)` → tri‑class via the **maybe band**.

### 3) LLM labeling (`llm_label.py`)
- Async OpenAI calls with strict prompts in `prompts.py` (MASLD/NAFLD RCT scope).
- Writes for each paper: `llm_label`, `llm_conf`, `llm_prob_include` (derived), `llm_reasons`, `llm_evidence_spans`.

### 4) Fusion (`fuse_and_predict.py`)
- Features: supervised probs **[exclude, maybe, include]**, LLM include‑prob, keyword heuristic, disagreement & entropy.
- Meta‑model: **LogisticRegression** (weighted toward `include` if desired).
- Outputs `outputs/final_predictions.parquet` with the final label and rich diagnostics.

### 5) Self‑training (`self_training.py`) *(optional)*
- Picks **high‑confidence** examples from fusion (`prob_include ≥ pseudo_min_prob` or ≪ for exclude).
- Retrains supervised model (with small weight for pseudo‑labels) and re‑calibrates on gold.

---

## Prompts & Study Scope

See `prompts.py`. The system and user instructions enforce:

- **Population**: Adults (≥18) with MASLD/NAFLD/fatty liver/steatosis; allow diabetes/obesity/HIV/metabolic syndrome.  
  Exclude NASH/fibrosis/HCC/cancer and pediatric.
- **Intervention**: Aims to reduce hepatic fat (not just glycemic control).
- **Comparison**: Comparator arm required (healthy/non‑MASLD, placebo, or another treatment).
- **Outcomes**: Liver steatosis/function measurements (MRI/US/CAP, enzymes/scores, insulin resistance/glucose/lipids).
- **Design**: **RCT** (quantitative).

When uncertain → **MAYBE**. LLM returns concise `reasons` and verbatim `evidence_spans` for auditability.

---

## Outputs

`outputs/final_predictions.parquet` includes (columns may expand over time):

- `pred_label`: **include / exclude / maybe** (from fusion)
- `prob_include`: final fused probability of include
- `sup_prob_include`: supervised model `P(include)`
- `llm_prob_include`: derived from LLM `label+confidence`
- `kw_rule`: keyword heuristic score \[0,1]
- `llm_label`, `llm_conf`, `llm_reasons`, `llm_evidence_spans`
- plus original text columns from `unl.parquet`

Load and inspect:

```python
import pandas as pd
df = pd.read_parquet("outputs/final_predictions.parquet")
df.sort_values("prob_include", ascending=False).head(10)
```

---

## Tuning for **INCLUDE** performance

- Prefer **binary include‑vs‑rest** training if your gold set is small.
- Increase `include_weight` in `ModelCfg` (and in fusion `class_weight`) to favor **include** recall.
- Keep calibration (isotonic) so thresholds mean what they say.
- Adjust `maybe_low/high` to control conservativeness around the decision boundary.
- Consider enabling `self_training.py` after a first pass to absorb confident new inclusions.

---

## Performance tips

- LightGBM: start with `num_leaves=31`, `max_depth=6`, `min_data_in_leaf=30`, `feature_fraction=bagging_fraction=0.8`, `reg_lambda≈1.0`.  
  Then tune `n_estimators` with early stopping (if you add a val set) and tweak regularization.
- If classes are imbalanced, you can add **SMOTE** in the supervised stage (see prior discussions) or keep `class_weight="balanced"`.
- For LLM: increase `concurrency` carefully; back off if rate-limited.

---

## Troubleshooting

**PyArrow dtype errors when writing parquet**  
Example: `ArrowInvalid: Could not convert '7' with type str...`  
Cause: mixed types in a column. Fix by casting in pandas before writing:
```python
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
```

**LightGBM warnings**: `No further splits with positive gain`  
Usually benign with low‑dim embeddings. You can increase `num_leaves` or relax regularization, but it’s often safe to ignore.

**scikit-learn class_weight error**:  
`ValueError: The classes, [2], are not in class_weight`  
Ensure your `class_weight` **keys match numeric labels** present in `y` (e.g., `{0:1.0, 2:3.0}` if your labels are `{0,2}`). Or use `"balanced"`.

**OpenAI rate limits / timeouts**  
Reduce `LLMCfg.concurrency`, increase `max_retries`, and keep `timeout_s` reasonable.

**Windows event loop**  
`llm_label.py` sets a Windows‑compatible loop policy; keep it if you are on Windows.

---

## Reproducibility

- Global RNG seed: `ModelCfg.random_state` (used in splits and models).  
- Determinism across CUDA/cuBLAS is not guaranteed if you use a GPU for embeddings.

---

## Security & Privacy

- This pipeline reads raw titles/abstracts and stores text in `artifacts/`.  
- Keep your `.env` out of version control (already in `.gitignore`).  
- Review outputs for sensitive content before sharing.

---

## License

MIT `LICENSE`.

---

