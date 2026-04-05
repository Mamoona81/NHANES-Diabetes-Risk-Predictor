from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _bootstrap_import_path() -> None:
    """Allow running as `python src/model.py` while importing `src.*`."""
    if __package__ is None or __package__ == "":
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root))


_bootstrap_import_path()

from src.data_loader import load_nhanes_2011_2018  # noqa: E402
from src.preprocessing import make_preprocessor, split_X_y_and_weights  # noqa: E402


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NHANES diabetes risk model (non-lab default).")
    p.add_argument("--with-labs", action="store_true", help="Include HbA1c (LBXGH) as a predictor.")
    p.add_argument("--cache-dir", type=str, default=str(Path("data") / "cache"), help="NHANES cache dir")
    p.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output directory")
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from joblib import dump
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier
    except ModuleNotFoundError as e:
        missing = str(e).split("No module named ")[-1].strip("'")
        raise SystemExit(
            f"Missing dependency: {missing}. Run: pip install -r requirements.txt"
        ) from e

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)

    df = load_nhanes_2011_2018(cache_dir=cache_dir, with_labs=args.with_labs)
    X, y, w = split_X_y_and_weights(df, with_labs=args.with_labs)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    w_train = w.iloc[train_idx] if w is not None else None
    w_test = w.iloc[test_idx] if w is not None else None

    pre = make_preprocessor(X_train)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    clf = XGBClassifier(
        n_estimators=700 if args.with_labs else 600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=0,
        scale_pos_weight=scale_pos_weight,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    fit_kwargs = {}
    if w_train is not None:
        fit_kwargs["clf__sample_weight"] = w_train

    pipe.fit(X_train, y_train, **fit_kwargs)

    p_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test, sample_weight=w_test)

    dump(pipe, artifacts_dir / "model.joblib")
    np.savez_compressed(artifacts_dir / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)

    meta = {
        "python_entry": "python src/model.py",
        "target": "DIQ010: 1 vs 2; borderline/unknown/refused dropped",
        "cycles": ["2011-2012", "2013-2014", "2015-2016", "2017-2018"],
        "with_labs": bool(args.with_labs),
        "test_auc": float(auc),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved {artifacts_dir / 'model.joblib'} | test AUROC={auc:.4f} | with_labs={args.with_labs}")


if __name__ == "__main__":
    main()
