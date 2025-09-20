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

def tri_probs_from_p_inc(p_inc, maybe_low, maybe_high):
    p = np.asarray(p_inc, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    p_exc = 1.0 - p

    center = (maybe_low + maybe_high) / 2.0
    halfw  = max(1e-6, (maybe_high - maybe_low) / 2.0)
    
    p_may = 1.0 - np.abs(p - center) / halfw
    p_may = np.clip(p_may, 0.0, 1.0)

    denom = p_exc + p_may + p
    p_exc /= denom
    p_may /= denom
    p     /= denom

    return np.column_stack([p_exc, p_may, p])

def save_numpy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_numpy(path: str) -> np.ndarray:
    return np.load(path)

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(float(x), hi))

def keyword_rule_score(text: str) -> float:
    """Heuristic: boost adult MASLD RCTs with comparator & liver outcomes; penalize reviews/protocols/non-human/pediatric/observational/NASH-fibrosis-HCC.
       Same logic as before: sum(term in text) and sigmoid."""
    t = (text or "").lower()

    # POSITIVE signals (presence adds +1 each)
    include_terms = [
        # RCT / control structure
        "randomized", "randomised", "random allocation", "double blind", "double-blind",
        "placebo-controlled", "controlled trial", "parallel group", "randomized controlled trial", "rct",
        # comparator cues
        "versus", " vs ", "compared with", "compared to", "control group", "placebo",
        "usual care", "standard care", "active comparator",
        # target disease
        "masld", "nafld", "nonalcoholic fatty liver disease", "fatty liver",
        "hepatic steatosis", "liver steatosis",
        # adult population
        "adult", "adults ",
        # outcomes (steatosis/function/metabolic markers commonly reported in NAFLD trials)
        "mri-pdff", "pdff", "magnetic resonance imaging", "ultrasound", "fibroscan",
        "controlled attenuation parameter",  # avoid bare "cap" to reduce false hits
        "alt", "ast", "ggt", "liver enzyme", "liver enzymes", "liver function", "liver function test",
        "homa-ir", "insulin resistance", "glucose", "triglyceride", "cholesterol"
    ]

    # NEGATIVE signals (presence adds +1 to exc; weighted by 0.8 below)
    exclude_terms = [
        # publication types not eligible
        "protocol", "study protocol", "editorial", "commentary", "letter",
        "systematic review", "meta-analysis", "scoping review", "narrative review", "review article",
        # non-human / lab-only
        "in vitro", "cell line", "cells ", "animal", "mouse", "mice", "rat", "murine", "rabbit", "porcine", "canine", "feline",
        # pediatric
        "children", "child", "paediatric", "pediatric", "adolescent", "teen", "infant", "neonate",
        # observational / uncontrolled designs
        "observational", "cohort", "cross-sectional", "case-control", "retrospective", "prospective cohort",
        "registry", "database study", "single-arm", "single arm", "before-after", "pre-post", "pretest-posttest", "uncontrolled",
        # excluded populations
        "nash", "nonalcoholic steatohepatitis", "fibrosis", "cirrhosis", "hepatocellular carcinoma", "hcc", "cancer"
    ]

    inc = sum(term in t for term in include_terms)
    exc = sum(term in t for term in exclude_terms)

    raw = inc - 0.8 * exc  # same weighting as your original
    # squash to [0,1]
    return float(1.0 / (1.0 + np.exp(-raw)))

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
