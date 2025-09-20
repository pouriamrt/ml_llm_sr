import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class Paths:
    data_dir: str = "data"
    labeled_exclude_csv: str = "data/labeled_exclude.csv"
    labeled_include_csv: str = "data/labeled_include.csv"
    labeled_maybe_csv: str = "data/labeled_maybe.csv"
    unlabeled_csv: str = "data/unlabeled_6000.csv"

    artifacts_dir: str = "artifacts"
    emb_dir: str = "artifacts/embeddings"
    sup_dir: str = "artifacts/supervised"
    llm_dir: str = "artifacts/llm"
    fusion_dir: str = "artifacts/fusion"

    outputs_dir: str = "outputs"

@dataclass
class ModelCfg:
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    columns: list[str] = field(default_factory=lambda: ["Key", "Title", "Abstract Note", "Manual Tags"])
    # supervised
    lightgbm_num_leaves: int = 31
    lightgbm_estimators: int = 2000
    lightgbm_lr: float = 0.04
    lightgbm_max_depth: int = 6
    lightgbm_min_data_in_leaf: int = 30
    lightgbm_feature_fraction: float = 0.8
    lightgbm_bagging_fraction: float = 0.8
    lightgbm_bagging_freq: int = 1
    random_state: int = 42
    maybe_low: float = 0.40
    maybe_high: float = 0.60
    # thresholds for auto-pseudolabeling (used in self_training.py)
    pseudo_min_prob: float = 0.975

@dataclass
class LLMCfg:
    model: str = "gpt-4.1-mini"
    api_key: str = os.getenv("OPENAI_API_KEY")
    temperature: float = 0.0
    max_retries: int = 3
    batch_size: int = 20  # title+abstract are short; safe small batches
    # mapping LLM textual confidence to numeric
    confidence_map = {"high": 0.90, "medium": 0.70, "low": 0.55}
    concurrency: int = 10
    timeout_s: float = 60.0
    use_fewshots: bool = False
    
LABEL_MAP = {"exclude": 0, "maybe": 1, "include": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def ensure_dirs(p: Paths):
    for d in [p.artifacts_dir, p.emb_dir, p.sup_dir, p.llm_dir, p.fusion_dir, p.outputs_dir, p.data_dir]:
        os.makedirs(d, exist_ok=True)