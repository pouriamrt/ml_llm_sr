import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

from config import Paths, ModelCfg, LABEL_MAP, ensure_dirs, INV_LABEL_MAP

def predict_with_maybe(clf, X, maybe_low, maybe_high):
    """
    Return (y_pred_labels, p_include) where y_pred_labels are in {exclude, maybe, include}
    even if the model was trained binary. Relies on calibrated probabilities.
    """
    proba = clf.predict_proba(X)  # (n, K)
    classes = clf.classes_         # e.g., [0, 2] for exclude/include

    # index of 'include' in clf.classes_
    try:
        inc_idx = int(np.where(classes == LABEL_MAP["include"])[0][0])
    except IndexError:
        raise ValueError("Include class not found in clf.classes_. Check LABEL_MAP and training labels.")

    p_inc = proba[:, inc_idx]

    # Start from exclude, then overwrite maybe/include by thresholds
    y_pred = np.full(shape=len(p_inc), fill_value=LABEL_MAP["exclude"], dtype=int)
    # maybe band
    maybe_mask = (p_inc >= maybe_low) & (p_inc <= maybe_high)
    y_pred[maybe_mask] = LABEL_MAP["maybe"]
    # include high-confidence
    y_pred[p_inc > maybe_high] = LABEL_MAP["include"]

    return y_pred, p_inc

def main():
    paths = Paths()
    cfg = ModelCfg()
    ensure_dirs(paths)

    Xg = np.load(os.path.join(paths.emb_dir, "gold.npy"))
    gold = pd.read_parquet(os.path.join(paths.emb_dir, "gold.parquet"))
    y  = gold["label"].map(LABEL_MAP).values
    
    uniq = np.unique(y)
    if len(uniq) == 1:
        raise ValueError("Only one label found, skipping training.")
    elif len(uniq) == 2:
        obj = "binary"
    else:
        obj = "multiclass"
    present_labels_sorted = sorted(uniq.tolist())
    target_names = [INV_LABEL_MAP[i] for i in present_labels_sorted]

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xg, y, test_size=0.2, stratify=y, random_state=cfg.random_state
    )

    base = LGBMClassifier(
        n_estimators=cfg.lightgbm_estimators,
        num_leaves=cfg.lightgbm_num_leaves,
        learning_rate=cfg.lightgbm_lr,
        objective=obj,
        max_depth=cfg.lightgbm_max_depth,
        class_weight="balanced",
        min_data_in_leaf=cfg.lightgbm_min_data_in_leaf,
        feature_fraction=cfg.lightgbm_feature_fraction,
        bagging_fraction=cfg.lightgbm_bagging_fraction,
        bagging_freq=cfg.lightgbm_bagging_freq,
        reg_lambda=1.0,
        reg_alpha=0.05,
        max_bin=255,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=-1
    )
    # Calibrate to get well-behaved probabilities
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_tr, y_tr)
    
    # (A) Standard argmax report using the trained label set
    y_pred_raw = clf.predict(X_te)
    print("=== Raw model (no maybe injection) ===")
    print(classification_report(y_te, y_pred_raw, target_names=target_names))

    # (B) If we trained BINARY, also show thresholded preds with 'maybe' injection
    if obj == "binary":
        y_pred_thr, p_inc = predict_with_maybe(clf, X_te, cfg.maybe_low, cfg.maybe_high)
        counts = pd.Series([INV_LABEL_MAP[i] for i in y_pred_thr]).value_counts()
        print("\n=== Thresholded predictions with 'maybe' injection (binary training) ===")
        print(f"P(include) band -> maybe when {cfg.maybe_low:.2f} ≤ p ≤ {cfg.maybe_high:.2f}")
        print("Predicted label distribution (thresholded):")
        print(counts.to_string())

        # report accuracy after collapsing 'maybe' to closest class for metrics ONLY
        # (this is purely for monitoring; keep 'maybe' in your final outputs)
        collapse = np.where(y_pred_thr == LABEL_MAP["maybe"],
                            np.where(p_inc > 0.5, LABEL_MAP["include"], LABEL_MAP["exclude"]),
                            y_pred_thr)
        print("\nAccuracy if we collapse 'maybe' to nearest class (monitoring only):")
        acc = (collapse == y_te).mean()
        print(f"{acc:.4f}")
        
    # Persist artifacts
    os.makedirs(paths.sup_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(paths.sup_dir, "supervised_clf.pkl"))
    print("Saved supervised model.")


if __name__ == "__main__":
    main()
