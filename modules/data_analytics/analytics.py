
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]


def analyze_dataset(file_path):
    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Load any CSV dataset
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    output_dir = BASE_DIR / "outputs" / "analytics" / file_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EXAMGUARD AI: UNIVERSAL DATASET ANALYZER ===")
    print("Dataset:", file_path.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Detect data types automatically
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols
    ]

    print("\n--- Detected Columns ---")
    print("Numeric columns:", numeric_cols)
    print("Other columns:", categorical_cols)

    # Data quality report
    quality = pd.DataFrame({
        "column": df.columns,
        "data_type": [str(df[col].dtype) for col in df.columns],
        "missing_values": [int(df[col].isna().sum()) for col in df.columns],
        "unique_values": [int(df[col].nunique()) for col in df.columns],
    })

    quality.to_csv(output_dir / "data_quality_report.csv", index=False)

    print("\n--- Data Quality ---")
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Total missing values:", int(df.isna().sum().sum()))
    print(quality.to_string(index=False))

    # Numeric statistics
    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats.to_csv(output_dir / "numeric_statistics.csv")

        print("\n--- Numeric Statistics ---")
        print(stats.round(2).to_string())

        # Find possible statistical outliers using IQR
        outlier_rows = []

        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.dropna()

            if values.empty:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask = (
                pd.to_numeric(df[col], errors="coerce").lt(lower)
                | pd.to_numeric(df[col], errors="coerce").gt(upper)
            )

            for index in df.index[mask.fillna(False)]:
                outlier_rows.append({
                    "row_number": int(df.index.get_loc(index)) + 2,
                    "column": col,
                    "value": df.loc[index, col],
                    "lower_bound": lower,
                    "upper_bound": upper,
                })

        pd.DataFrame(
            outlier_rows,
            columns=[
                "row_number", "column", "value",
                "lower_bound", "upper_bound"
            ],
        ).to_csv(output_dir / "possible_outliers.csv", index=False)

        # Generate numeric distribution charts
        chart_cols = numeric_cols[:6]
        df[chart_cols].hist(figsize=(12, 8), bins=10)
        plt.tight_layout()
        plt.savefig(output_dir / "numeric_distributions.png")
        plt.close()

    # Summarize non-numeric columns
    if categorical_cols:
        with open(
            output_dir / "category_summaries.txt",
            "w",
            encoding="utf-8",
        ) as report:
            for col in categorical_cols:
                report.write(f"\nColumn: {col}\n")
                report.write(
                    df[col].value_counts(dropna=False)
                    .head(20)
                    .to_string()
                )
                report.write("\n")

    # Save a copy of the original data for traceability
    df.to_csv(output_dir / "dataset_copy.csv", index=False)

    print("\nReports saved to:", output_dir)
    print("Analysis completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze different CSV datasets automatically."
    )
    parser.add_argument(
        "file",
        help="Path to the CSV dataset, relative to the project folder "
             "or an absolute path",
    )
    args = parser.parse_args()

    analyze_dataset(args.file)
