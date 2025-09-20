import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from train_supervised import predict_with_maybe
from config import Paths, LABEL_MAP, INV_LABEL_MAP, ensure_dirs, ModelCfg
from utils import keyword_rule_score, clamp, tri_probs_from_p_inc

def build_features(base_proba: np.ndarray, llm_prob_inc: np.ndarray, texts: pd.Series) -> np.ndarray:
    """Stack features for meta-model: supervised probs + LLM include prob + heuristics."""
    # base_proba shape (n, 3): [exclude, maybe, include]
    p_exc = base_proba[:,0]
    p_may = base_proba[:,1]
    p_inc = base_proba[:,2]
    # heuristics
    kw = np.array([keyword_rule_score(t) for t in texts], dtype=float)
    # disagreement signals
    diff_inc = np.abs(p_inc - llm_prob_inc)
    ent_model = -(p_exc*np.log(p_exc+1e-9) + p_may*np.log(p_may+1e-9) + p_inc*np.log(p_inc+1e-9))
    # feature matrix
    X = np.vstack([p_exc, p_may, p_inc, llm_prob_inc, kw, diff_inc, ent_model]).T
    return X

def main():
    paths = Paths()
    cfg = ModelCfg()
    ensure_dirs(paths)

    # Load supervised model
    sup = joblib.load(os.path.join(paths.sup_dir, "supervised_clf.pkl"))

    # GOLD (to train meta-fuser)
    Xg = np.load(os.path.join(paths.emb_dir, "gold.npy"))
    gold = pd.read_parquet(os.path.join(paths.emb_dir, "gold.parquet"))
    y = gold["label"].map(LABEL_MAP).values
    
    include_weight = getattr(cfg, "include_weight", 4.0)
    uniq = np.unique(y)
    if len(uniq) == 1:
        raise ValueError("Only one label found, skipping training.")
    elif len(uniq) == 2:
        _, p_inc = predict_with_maybe(sup, Xg, cfg.maybe_low, cfg.maybe_high)
        proba_g = tri_probs_from_p_inc(p_inc, cfg.maybe_low, cfg.maybe_high)
        col_idx = 1
        class_weight = {0: 1.0, 2: float(include_weight)}
    else:
        proba_g = sup.predict_proba(Xg)
        col_idx = LABEL_MAP["include"]
        class_weight = {0: 1.0, 1: 1.0, 2: float(include_weight)}
    
    present_labels_sorted = sorted(uniq.tolist())
    target_names = [INV_LABEL_MAP[i] for i in present_labels_sorted]

    llm_g = pd.read_parquet(os.path.join(paths.llm_dir, "llm_labels_gold.parquet"))
    assert len(llm_g) == len(gold), "Mismatch gold vs LLM rows"

    X_meta_g = build_features(proba_g, llm_g["llm_prob_include"].values, gold["text"])

    # Train meta on gold
    X_tr, X_te, y_tr, y_te = train_test_split(X_meta_g, y, test_size=0.2, stratify=y, random_state=cfg.random_state)
    meta = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=cfg.random_state)
    meta.fit(X_tr, y_tr)

    y_pred = meta.predict(X_te)
    y_proba = meta.predict_proba(X_te)[:, col_idx]
    print(classification_report(y_te, y_pred, target_names=target_names))
    if len(uniq) == 2:
        y_pred_thr, _ = predict_with_maybe(meta, X_te, cfg.maybe_low, cfg.maybe_high)
        counts = pd.Series([INV_LABEL_MAP[i] for i in y_pred_thr]).value_counts()
        print("\n=== Thresholded predictions with 'maybe' injection (binary training) ===")
        print(f"P(include) band -> maybe when {cfg.maybe_low:.2f} ≤ p ≤ {cfg.maybe_high:.2f}")
        print("Predicted label distribution (thresholded):")
        print(counts.to_string())
        
    try:
        print("Binary AUROC (include vs rest):", roc_auc_score((y_te==LABEL_MAP["include"]).astype(int), y_proba))
    except Exception:
        pass

    # Save meta
    os.makedirs(paths.fusion_dir, exist_ok=True)
    joblib.dump(meta, os.path.join(paths.fusion_dir, "meta_fuser.pkl"))

    # PREDICT on UNLABELED
    Xu = np.load(os.path.join(paths.emb_dir, "unl.npy"))
    unl = pd.read_parquet(os.path.join(paths.emb_dir, "unl.parquet"))
    if len(uniq) == 2:
        _, p_inc = predict_with_maybe(sup, Xu, cfg.maybe_low, cfg.maybe_high)
        proba_u = tri_probs_from_p_inc(p_inc, cfg.maybe_low, cfg.maybe_high)
    else:
        proba_u = sup.predict_proba(Xu)
        
    llm_u = pd.read_parquet(os.path.join(paths.llm_dir, "llm_labels_unl.parquet"))
    assert len(llm_u) == len(unl), "Mismatch unl vs LLM rows"

    X_meta_u = build_features(proba_u, llm_u["llm_prob_include"].values, unl["text"])
    if len(uniq) == 2:
        _, p_inc = predict_with_maybe(meta, X_meta_u, cfg.maybe_low, cfg.maybe_high)
        meta_proba_u = tri_probs_from_p_inc(p_inc, cfg.maybe_low, cfg.maybe_high)
    else:
        meta_proba_u = meta.predict_proba(X_meta_u)

    # Final decision = argmax meta class
    pred_idx = meta_proba_u.argmax(axis=1)
    pred_label = [INV_LABEL_MAP[i] for i in pred_idx]
    p_inc = meta_proba_u[:, LABEL_MAP["include"]]

    out = unl.copy()
    out["pred_label"] = pred_label
    out["prob_include"] = p_inc
    out["sup_prob_include"] = proba_u[:, LABEL_MAP["include"]]
    out["llm_prob_include"] = llm_u["llm_prob_include"].values
    out["kw_rule"] = [keyword_rule_score(t) for t in unl["text"]]
    out["llm_label"] = llm_u["llm_label"].values
    out["llm_conf"] = llm_u["llm_conf"].values
    out["llm_reasons"] = llm_u["llm_reasons"].values
    out["llm_evidence_spans"] = llm_u["llm_evidence_spans"].values
    out["decision_sources"] = "meta(sup+llm+kw)"

    os.makedirs(paths.outputs_dir, exist_ok=True)
    out_path = os.path.join(paths.outputs_dir, "final_predictions.parquet")
    out.reset_index(drop=True).to_parquet(out_path)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
