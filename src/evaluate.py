from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _bootstrap_import_path() -> None:
    if __package__ is None or __package__ == "":
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root))


_bootstrap_import_path()

from src.data_loader import load_nhanes_2011_2018  # noqa: E402
from src.preprocessing import split_X_y_and_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved model (metrics + fairness + SHAP).")
    p.add_argument("--artifacts-dir", type=str, default="artifacts")
    p.add_argument("--cache-dir", type=str, default=str(Path("data") / "cache"))
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        import shap
        from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
    except ModuleNotFoundError as e:
        missing = str(e).split("No module named ")[-1].strip("'")
        raise SystemExit(
            f"Missing dependency: {missing}. Run: pip install -r requirements.txt"
        ) from e

    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipe = load(artifacts_dir / "model.joblib")

    meta_path = artifacts_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())

    with_labs = bool(meta.get("with_labs", False))

    df = load_nhanes_2011_2018(cache_dir=Path(args.cache_dir), with_labs=with_labs)
    X, y, w = split_X_y_and_weights(df, with_labs=with_labs)

    split = np.load(artifacts_dir / "split_indices.npz")
    test_idx = split["test_idx"].astype(int)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    w_test = w.iloc[test_idx].reset_index(drop=True) if w is not None else None

    p = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, p, sample_weight=w_test)
    ap = average_precision_score(y_test, p, sample_weight=w_test)
    brier = brier_score_loss(y_test, p, sample_weight=w_test)

    print(f"AUROC={auc:.4f}  AUPRC={ap:.4f}  Brier={brier:.4f}  with_labs={with_labs}")

    y_hat = (p >= float(args.threshold)).astype(int)

    # Fairness: group metrics by sex and race/ethnicity if present
    sensitive = {}
    if "RIAGENDR" in X_test.columns:
        sensitive["sex"] = X_test["RIAGENDR"]
    if "RIDRETH3" in X_test.columns:
        sensitive["race_eth"] = X_test["RIDRETH3"]

    for name, s in sensitive.items():
        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_test,
            y_pred=pd.Series(y_hat),
            sensitive_features=s,
        )
        print(f"\nGroup metrics by {name} (threshold={args.threshold}):")
        print(mf.by_group)

    # SHAP summary plot
    X_test_trans = pipe.named_steps["pre"].transform(X_test)
    feature_names = pipe.named_steps["pre"].get_feature_names_out()

    n = min(2000, X_test_trans.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test_trans.shape[0], size=n, replace=False)
    Xs = X_test_trans[idx]

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(Xs)

    plt.figure()
    shap.summary_plot(shap_values, features=Xs, feature_names=feature_names, show=False)
    plt.tight_layout()
    out_path = reports_dir / "shap_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
