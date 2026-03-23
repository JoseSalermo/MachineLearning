from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from salary_service.modeling import save_artifacts, train_final_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the salary model.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a CSV file with the training data.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the trained model and metadata will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

    df = pd.read_csv(dataset_path)
    candidate_model, metrics = train_model(df)
    final_model = train_final_model(df)
    save_artifacts(final_model, args.output_dir, metrics=metrics)

    print(f"Saved model artifacts to {Path(args.output_dir).resolve()}")
    print(
        "Holdout metrics "
        f"MAE={metrics.mae:,.2f} RMSE={metrics.rmse:,.2f} R2={metrics.r2:.4f}"
    )
    print(f"Candidate model tree count: {candidate_model.tree_count_}")
    print(f"Final model tree count: {final_model.tree_count_}")


if __name__ == "__main__":
    main()
