"""
Confluence — Step 1: Data Pipeline
===================================
Loads PaySim (6.3M real transaction rows), makes it realistically messy,
cleans it with pandas / cudf.pandas, and outputs a benchmark comparison.

PaySim dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
Place CSV at: ../data/PS_20174392719_1491204439457_log.csv

Run in Colab with a GPU runtime for real benchmark numbers.
"""

import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── GPU Detection ────────────────────────────────────────────────────────────
GPU_AVAILABLE = False
try:
    import cudf.pandas
    cudf.pandas.install()
    GPU_AVAILABLE = True
    print("✅ cudf.pandas active — GPU acceleration ENABLED")
except ImportError:
    print("⚠️  cudf not found — running on CPU pandas (expected on local/Windows)")
    print("   On Colab GPU: !pip install cudf-cu12 --extra-index-url https://pypi.nvidia.com")

RNG = np.random.default_rng(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH  = "/content/PS_20174392719_1491204439457_log.csv"
OUTPUT_DIR = "/content/outputs"


# ─── 1. LOAD PAYSIM ───────────────────────────────────────────────────────────

def load_paysim(path: str) -> pd.DataFrame:
    """
    Load PaySim CSV and normalize into the schema used throughout Confluence:
        transaction_id | sender_id | receiver_id | amount | timestamp | tx_type | is_fraud
    """
    print(f"\nLoading PaySim from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # PaySim columns: step, type, amount, nameOrig, oldbalanceOrg,
    #                 newbalanceOrig, nameDest, oldbalanceDest,
    #                 newbalanceDest, isFraud, isFlaggedFraud

    # 'step' = hour number (1 step = 1 hour from start)
    base_time = pd.Timestamp("2024-01-01")
    df["timestamp"] = base_time + pd.to_timedelta(df["step"], unit="h")
    df["transaction_id"] = "TXN" + df.index.astype(str).str.zfill(8)

    df = df.rename(columns={
        "nameOrig":  "sender_id",
        "nameDest":  "receiver_id",
        "type":      "tx_type",
        "isFraud":   "is_fraud",
    })

    df = df[[
        "transaction_id", "sender_id", "receiver_id",
        "amount", "timestamp", "tx_type", "is_fraud"
    ]]

    print(f"  Normalized shape: {df.shape}")
    print(f"  Labeled fraud rows: {df['is_fraud'].sum():,} "
          f"({df['is_fraud'].mean()*100:.3f}%)")
    return df


# ─── 2. INJECT REALISTIC MESS ─────────────────────────────────────────────────

def inject_mess(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add the kinds of data quality issues real pipelines encounter.
    This makes the cleaning step non-trivial — and makes the cuDF
    benchmark meaningful, because there's real string + type work to do.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # 1. Inconsistent account ID casing + trailing whitespace (~1%)
    idx = rng.choice(n, size=n // 100, replace=False)
    df.loc[idx, "sender_id"] = df.loc[idx, "sender_id"].str.lower() + "  "

    # 2. Amounts stored as currency strings (~0.5%)
    df["amount"] = df["amount"].astype(object)
    idx2 = rng.choice(n, size=n // 200, replace=False)
    df.loc[idx2, "amount"] = df.loc[idx2, "amount"].apply(
        lambda x: f"${float(x):,.2f}"
    )

    # 3. Duplicate rows (~0.3%)
    dup_idx = rng.choice(n, size=n // 300, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 4. A handful of null amounts
    null_idx = rng.choice(len(df), size=50, replace=False)
    df.loc[null_idx, "amount"] = None

    print(f"  After injecting mess: {len(df):,} rows "
          f"(added {len(df) - n:,} duplicates + noise)")
    return df


# ─── 3. INJECT SYNTHETIC MULE RINGS on top of PaySim ─────────────────────────

def inject_mule_rings(df: pd.DataFrame, n_rings: int = 6) -> tuple[pd.DataFrame, list]:
    """
    Plant a small number of synthetic circular-flow mule rings into the dataset.
    We pick real account IDs that already exist (so graph connectivity is natural)
    and add transactions that form a closed loop with fast succession + high amounts.

    Returns updated df + a list of ring account groups (ground truth for eval).
    """
    existing_accounts = pd.unique(df["sender_id"].str.strip().str.upper())
    rng = np.random.default_rng(99)
    ring_ground_truth = []
    new_rows = []
    base_time = pd.Timestamp("2024-02-15")

    for ring_idx in range(n_rings):
        ring_size = int(rng.integers(4, 7))
        ring_accounts = rng.choice(existing_accounts, size=ring_size, replace=False).tolist()
        base_amount = float(rng.uniform(30000, 100000))
        t0 = base_time + pd.Timedelta(days=int(rng.integers(0, 20)))

        for hop in range(ring_size):
            new_rows.append({
                "transaction_id": f"RING{ring_idx:03d}HOP{hop:02d}",
                "sender_id":      ring_accounts[hop],
                "receiver_id":    ring_accounts[(hop + 1) % ring_size],
                "amount":         round(base_amount * (0.97 ** hop), 2),
                "timestamp":      t0 + pd.Timedelta(minutes=int(hop * rng.integers(10, 60))),
                "tx_type":        "TRANSFER",
                "is_fraud":       1,
            })
        ring_ground_truth.append(set(ring_accounts))

    injected_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"  Injected {n_rings} synthetic mule rings "
          f"({sum(len(r) for r in ring_ground_truth)} accounts involved)")
    return injected_df, ring_ground_truth


# ─── 4. CLEANING ──────────────────────────────────────────────────────────────

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize IDs, parse messy amounts, drop duplicates/nulls.
    Written in plain pandas — runs unchanged under cudf.pandas on GPU.
    """
    df = df.copy()

    # Normalize account IDs
    df["sender_id"]   = df["sender_id"].astype(str).str.strip().str.upper()
    df["receiver_id"] = df["receiver_id"].astype(str).str.strip().str.upper()

    # Parse messy amounts
    def parse_amount(val):
        if isinstance(val, str):
            return float(val.replace("$", "").replace(",", "").strip())
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    df["amount"] = df["amount"].apply(parse_amount)

    # Drop nulls + duplicates
    df = df.dropna(subset=["amount", "sender_id", "receiver_id"])
    df = df.drop_duplicates(
        subset=["sender_id", "receiver_id", "amount", "timestamp"]
    )
    df = df.reset_index(drop=True)
    return df


def benchmark_cleaning(messy_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Time the cleaning step. For a real CPU vs GPU comparison:
      - Run this script once WITHOUT cudf.pandas → record elapsed
      - Run again WITH cudf.pandas on a GPU runtime → record elapsed
      - The ratio is your benchmark slide number
    """
    print(f"\n  Cleaning {len(messy_df):,} rows ...")
    start = time.perf_counter()
    clean_df = clean_transactions(messy_df)
    elapsed = time.perf_counter() - start
    backend = "GPU (cudf.pandas)" if GPU_AVAILABLE else "CPU (pandas)"
    print(f"  ✅ [{backend}] {len(messy_df):,} rows → {len(clean_df):,} clean rows "
          f"in {elapsed:.3f}s")
    return clean_df, elapsed


# ─── 5. MAIN ──────────────────────────────────────────────────────────────────

def run_data_pipeline():
    print("=" * 65)
    print("CONFLUENCE — Step 1: Data Pipeline")
    print("=" * 65)

    # Load
    print("\n[1/4] Loading PaySim dataset")
    raw_df = load_paysim(DATA_PATH)

    # Inject mess
    print("\n[2/4] Injecting realistic data quality issues")
    messy_df = inject_mess(raw_df)

    # Inject synthetic mule rings (ground truth)
    print("\n[3/4] Planting synthetic mule rings for ground-truth eval")
    messy_df, ring_ground_truth = inject_mule_rings(messy_df)

    # Clean + benchmark
    print("\n[4/4] Cleaning (benchmarkable step)")
    clean_df, elapsed = benchmark_cleaning(messy_df)

    # Save
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_df.to_parquet(f"{OUTPUT_DIR}/clean_transactions.parquet", index=False)
    print(f"\n  Saved → {OUTPUT_DIR}/clean_transactions.parquet")

    # Save ring ground truth for evaluation in step 2
    import json
    ring_gt_serializable = [list(r) for r in ring_ground_truth]
    with open(f"{OUTPUT_DIR}/ring_ground_truth.json", "w") as f:
        json.dump(ring_gt_serializable, f)
    print(f"  Saved → {OUTPUT_DIR}/ring_ground_truth.json")

    # Benchmark record
    benchmark = {
        "backend": "GPU (cudf.pandas)" if GPU_AVAILABLE else "CPU (pandas)",
        "rows_processed": len(messy_df),
        "clean_rows": len(clean_df),
        "cleaning_time_sec": round(elapsed, 4),
    }
    with open(f"{OUTPUT_DIR}/benchmark_step1.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"  Saved → {OUTPUT_DIR}/benchmark_step1.json")
    print(f"\n  ⚡ Benchmark: {benchmark['backend']} — {elapsed:.3f}s for "
          f"{len(messy_df):,} rows")

    return clean_df, ring_ground_truth, elapsed


if __name__ == "__main__":
    clean_df, ring_ground_truth, elapsed = run_data_pipeline()
    print("\n--- Sample cleaned transactions ---")
    print(clean_df.head(5).to_string())
    print(f"\nFraud rows retained: {clean_df['is_fraud'].sum():,}")
