
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "online"
    / "raw"
    / "Students suspicious behaviors detection dataset_V1.csv"
)

OUTPUT_DIR = BASE_DIR / "datasets" / "online" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "exam_behavior_cleaned.csv"


def clean_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("\n=== EXAMGUARD AI: DATA CLEANING ===")
    print("Original rows:", len(df))
    print("Original columns:", len(df.columns))

    # Remove only exact duplicate rows.
    # Keep a record of how many were removed.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    print("Exact duplicate rows removed:", duplicate_count)

    # Validate labels before saving.
    if df["label"].isna().any():
        raise ValueError("Target label contains missing values.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected label values found.")

    # Preserve missing categorical values for now.
    # Model preprocessing will handle them later.
    missing_before = int(df.isna().sum().sum())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Missing values retained for preprocessing:", missing_before)
    print("Cleaned rows:", len(df))
    print("Cleaned columns:", len(df.columns))

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nRemaining missing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\nCleaned dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    clean_dataset()
