import numpy as np
import pandas as pd
import os
from typing import List, Dict

def prep_text_df(df: pd.DataFrame) -> pd.DataFrame:
    def join_title_abs(row):
        t = str(row.get("Title") or "").strip()
        a = str(row.get("Abstract Note") or "").strip()
        k = str(row.get("Manual Tags") or "").replace("*", "").strip()
        return (t + "\n" + a + "\n" + k).strip()
    df = df.copy()
    df["text"] = df.apply(join_title_abs, axis=1)
    return df

def save_numpy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_numpy(path: str) -> np.ndarray:
    return np.load(path)

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(float(x), hi))

def keyword_rule_score(text: str) -> float:
    """Optional heuristic: boost likely-include signals and penalize common excludes."""
    t = text.lower()
    include_terms = ["randomized", "randomised", "double-blind", "trial", "cohort", "longitudinal", "case-control", "rct"]
    exclude_terms = ["protocol", "editorial", "commentary", "review", "systematic review", "meta-analysis", "animal", "in vitro", "letter"]
    inc = sum(term in t for term in include_terms)
    exc = sum(term in t for term in exclude_terms)
    raw = inc - 0.8*exc
    # squashing to [0,1]
    return 1/(1+np.exp(-raw))

def get_unique_keywords(df: pd.DataFrame) -> List[str]:
    keywords = df['Manual Tags'].dropna().values
    unique_keywords = set()
    for k in keywords:
        for kk in k.split(";"):
            word = kk.replace("*", "").strip()
            if len(word) > 1:
                unique_keywords.add(word)

    unique_keywords = list(unique_keywords)
    print(f"Number of unique keywords: {len(unique_keywords)}")
    return unique_keywords
