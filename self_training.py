import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

from config import Paths, ModelCfg, LABEL_MAP

def main():
    paths = Paths()
    cfg = ModelCfg()

    # load existing supervised and meta predictions
    sup = joblib.load(os.path.join(paths.sup_dir, "supervised_clf.pkl"))

    Xg = np.load(os.path.join(paths.emb_dir, "gold.npy"))
    gold = pd.read_parquet(os.path.join(paths.emb_dir, "gold.parquet"))
    yg = gold["label"].map(LABEL_MAP).values

    # From fusion step output
    pred = pd.read_parquet(os.path.join(paths.outputs_dir, "final_predictions.parquet"))

    # select high-confidence pseudo-labels from meta
    mask_inc = pred["prob_include"] >= cfg.pseudo_min_prob
    mask_exc = (1 - pred["prob_include"]) >= cfg.pseudo_min_prob  # very confident exclude
    pseudo = pred[mask_inc | mask_exc].copy()
    pseudo_y = np.where(mask_inc.loc[pseudo.index], LABEL_MAP["include"], LABEL_MAP["exclude"])

    # combine embeddings
    Xu = np.load(os.path.join(paths.emb_dir, "unl.npy"))
    X_pseudo = Xu[pseudo.index.values]

    # simple re-train: concatenate gold + pseudo with small sample weight for pseudo
    X_all = np.vstack([Xg, X_pseudo])
    y_all = np.concatenate([yg, pseudo_y])

    uniq = np.unique(y_all)
    if len(uniq) == 1:
        raise ValueError("Only one label found, skipping training.")
    elif len(uniq) == 2:
        obj = "binary"
    else:
        obj = "multiclass"

    base = LGBMClassifier(
        n_estimators=cfg.lightgbm_estimators,
        num_leaves=cfg.lightgbm_num_leaves,
        learning_rate=cfg.lightgbm_lr,
        objective=obj,
        class_weight="balanced",
        random_state=cfg.random_state,
        max_depth=cfg.lightgbm_max_depth,
        min_data_in_leaf=cfg.lightgbm_min_data_in_leaf,
        feature_fraction=cfg.lightgbm_feature_fraction,
        bagging_fraction=cfg.lightgbm_bagging_fraction,
        bagging_freq=cfg.lightgbm_bagging_freq,
        lambda_l2=1.0,
        lambda_l1=0.0,
        max_bin=255,
        n_jobs=-1,
        verbosity=-1
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    # scikit-learn sample_weight only at fit-time of base; CalibratedClassifierCV quirks exist.
    # For simplicity, we fit base with weights then calibrate freshly.
    base.fit(X_all, y_all, sample_weight=np.concatenate([np.ones_like(yg, dtype=float), 0.2*np.ones_like(pseudo_y, dtype=float)]))
    # Calibrate on gold only to avoid leaking pseudo into calibration
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(Xg, yg, test_size=0.2, stratify=yg, random_state=cfg.random_state)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)

    os.makedirs(paths.sup_dir, exist_ok=True)
    joblib.dump(cal, os.path.join(paths.sup_dir, "supervised_clf.pkl"))
    print("Saved updated supervised model with pseudo-labels.")

if __name__ == "__main__":
    main()
