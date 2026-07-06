"""
Confluence — Step 3: Liquidity Forecast
=========================================
Two-pass liquidity-stress scoring for individual accounts.

Pass 1 — Heuristic (v1):
  Computes a gap_risk_score from the ratio of predicted outflow to inflow.
  Fast, interpretable, no training required.
  FILTERED: accounts need >= min_transactions total transactions to be
  scored — most PaySim customer accounts appear only once (as sender OR
  receiver, never both), which mathematically maxes gap_risk_score with
  no real liquidity signal behind it. This filter removes that noise.

Pass 2 — ML (v2):
  Fits a per-account LinearRegression on historical daily flow to project
  the next 7 days. Falls back to heuristic if cuML is unavailable or
  insufficient history exists.

Output:
  outputs/liquidity_output.csv    — entity-level risk table
  outputs/benchmark_step3.json    — timing metrics
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── GPU / CPU toggle ─────────────────────────────────────────────────────────
try:
    import cuml
    from cuml.linear_model import LinearRegression
    GPU_ML = True
    print("✅ cuML active — GPU ML acceleration ENABLED")
except ImportError:
    from sklearn.linear_model import LinearRegression
    GPU_ML = False
    print("⚠️  cuML not found — using scikit-learn (CPU)")

OUTPUT_DIR = "/content/outputs"
FORECAST_DAYS = 7
MIN_TRANSACTIONS = 3   # accounts need at least this many total txns to be scored


# ─── 1. LOAD ──────────────────────────────────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, "clean_transactions.parquet")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} transactions")
    return df


# ─── 2. BUILD DAILY FLOW TABLE ────────────────────────────────────────────────

def build_daily_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (account, date) pair, compute:
      - total_inflow   : sum of amounts received
      - total_outflow  : sum of amounts sent
      - net_flow       : inflow - outflow
    Returns a long-format DataFrame indexed by (account_id, date).
    """
    tx = df.copy()
    if "timestamp" not in tx.columns:
        tx["timestamp"] = pd.to_datetime(tx.get("step", 0), unit="h",
                                          origin="2023-01-01")
    tx["date"] = tx["timestamp"].dt.normalize()

    inflows = (
        tx.groupby(["receiver_id", "date"])["amount"]
          .sum()
          .reset_index()
          .rename(columns={"receiver_id": "account_id", "amount": "inflow"})
    )
    outflows = (
        tx.groupby(["sender_id", "date"])["amount"]
          .sum()
          .reset_index()
          .rename(columns={"sender_id": "account_id", "amount": "outflow"})
    )

    daily = pd.merge(inflows, outflows, on=["account_id", "date"], how="outer").fillna(0)
    daily["net_flow"] = daily["inflow"] - daily["outflow"]
    daily = daily.sort_values(["account_id", "date"]).reset_index(drop=True)
    return daily


# ─── 3. HEURISTIC FORECAST (v1) — NOW WITH ACTIVITY FILTER ───────────────────

def heuristic_liquidity_forecast(
    daily_df: pd.DataFrame,
    transaction_counts: pd.Series = None,
    min_transactions: int = MIN_TRANSACTIONS,
) -> tuple:
    """
    Scores each account by the ratio of mean outflow to mean inflow over
    their entire history. gap_risk_score = 1 means pure outflow (maximum
    risk); gap_risk_score = 0 means inflow always exceeds outflow.

    FILTER: accounts need at least `min_transactions` total transactions
    (as sender OR receiver, across the whole raw dataset) to be scored.
    Most PaySim customer accounts fail this — they appear exactly once —
    which is the point: a single one-off transaction gives no real
    liquidity signal, it just mathematically maxes the score at 1.0.
    """
    start = time.perf_counter()

    if transaction_counts is not None:
        eligible_accounts = transaction_counts[transaction_counts >= min_transactions].index
        before = daily_df["account_id"].nunique()
        daily_df = daily_df[daily_df["account_id"].isin(eligible_accounts)]
        print(f"  Filtered to {daily_df['account_id'].nunique():,} accounts with >= "
              f"{min_transactions} total transactions (from {before:,} total)")
    else:
        print("  WARNING: no transaction_counts provided — scoring ALL accounts, "
              "including one-off accounts with no real signal")

    if daily_df.empty:
        print("  WARNING: filter removed all accounts. Consider lowering "
              "min_transactions, or switch to merchant-level scoring (see README note).")
        empty = pd.DataFrame(columns=[
            "account_id", "mean_inflow", "mean_outflow", "total_inflow",
            "total_outflow", "n_days", "gap_risk_score", "pred_inflow",
            "pred_outflow", "entity_id", "entity_type", "model"
        ])
        return empty, time.perf_counter() - start

    summary = daily_df.groupby("account_id").agg(
        mean_inflow=("inflow",   "mean"),
        mean_outflow=("outflow", "mean"),
        total_inflow=("inflow",  "sum"),
        total_outflow=("outflow","sum"),
        n_days=("date",         "count"),
    ).reset_index()

    total = summary["mean_inflow"] + summary["mean_outflow"]
    summary["gap_risk_score"] = np.where(
        total > 0,
        summary["mean_outflow"] / total,
        0.5,
    )
    summary["gap_risk_score"] = summary["gap_risk_score"].clip(0, 1)

    summary["pred_inflow"]  = summary["mean_inflow"]  * FORECAST_DAYS
    summary["pred_outflow"] = summary["mean_outflow"] * FORECAST_DAYS
    summary["entity_id"]    = summary["account_id"]
    summary["entity_type"]  = "account"
    summary["model"]        = "heuristic_v1"

    n_at_max = (summary["gap_risk_score"] >= 0.999).sum()
    print(f"  Scored {len(summary):,} eligible accounts "
          f"({n_at_max:,} still sit at the score ceiling of 1.0)")

    elapsed = time.perf_counter() - start
    return summary, elapsed


# ─── 4. ML FORECAST (v2) ──────────────────────────────────────────────────────

def ml_liquidity_forecast(daily_df: pd.DataFrame) -> tuple:
    """
    Per-account LinearRegression on day_number → outflow.
    Only accounts with >= 2 data points are modelled; the rest are skipped
    (this already naturally excludes true one-off accounts for the ML pass).
    """
    start = time.perf_counter()
    records = []
    accounts = daily_df["account_id"].unique()
    backend = "GPU (cuML)" if GPU_ML else "CPU (sklearn)"

    print(f"  Training linear regression models per account ...")

    for acct in accounts:
        sub = daily_df[daily_df["account_id"] == acct].copy()
        if len(sub) < 2:
            continue

        min_date = sub["date"].min()
        sub["day_num"] = (sub["date"] - min_date).dt.days

        X = sub[["day_num"]].values
        y_out = sub["outflow"].values
        y_in  = sub["inflow"].values

        try:
            reg_out = LinearRegression().fit(X, y_out)
            reg_in  = LinearRegression().fit(X, y_in)

            next_day = np.array([[sub["day_num"].max() + FORECAST_DAYS]])
            pred_out = max(0, float(reg_out.predict(next_day)[0]))
            pred_in  = max(0, float(reg_in.predict(next_day)[0]))

            total = pred_in + pred_out
            score = (pred_out / total) if total > 0 else 0.5

            records.append({
                "entity_id":      acct,
                "entity_type":    "account",
                "model":          f"ml_v2_{backend.split()[0].lower()}",
                "pred_inflow":    round(pred_in,  2),
                "pred_outflow":   round(pred_out, 2),
                "gap_risk_score": round(min(1.0, score), 4),
            })
        except Exception:
            continue

    elapsed = time.perf_counter() - start
    ml_df = pd.DataFrame(records)
    print(f"  [{backend}] Trained {len(ml_df)} account models in {elapsed:.3f}s")
    return ml_df, elapsed


# ─── 5. RISK FLAGGING ─────────────────────────────────────────────────────────

def flag_high_risk(df: pd.DataFrame,
                   high_threshold: float = 0.75,
                   med_threshold:  float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["alert_level"] = "LOW"
    df.loc[df["gap_risk_score"] >= med_threshold,  "alert_level"] = "MEDIUM"
    df.loc[df["gap_risk_score"] >= high_threshold, "alert_level"] = "HIGH"
    return df


# ─── 6. MAIN ──────────────────────────────────────────────────────────────────

def run_liquidity_forecast():
    print("=" * 65)
    print("CONFLUENCE — Step 3: Liquidity Forecast")
    print("=" * 65)

    df = load_transactions()

    # Total transaction count per account (sender OR receiver appearances)
    tx_counts = pd.concat([df["sender_id"], df["receiver_id"]]).value_counts()
    print(f"  Transaction count distribution — accounts with only 1 txn: "
          f"{(tx_counts == 1).sum():,} of {len(tx_counts):,} total accounts")

    # [1/3] Daily flow table
    print("\n[1/3] Building daily flow table")
    daily_df = build_daily_flow_table(df)
    print(f"  Daily flow table: {len(daily_df):,} rows (accounts × active days)")

    # [2/3] Heuristic (now filtered)
    print("\n[2/3] Heuristic forecast (v1)")
    heuristic_df, h_elapsed = heuristic_liquidity_forecast(daily_df, tx_counts)
    print(f"  Heuristic complete in {h_elapsed:.3f}s")

    # [3/3] ML forecast on top-500 accounts by outflow
    print("\n[3/3] ML forecast (v2)")
    if heuristic_df.empty:
        print("  Skipping ML — no eligible accounts from heuristic pass")
        ml_df, ml_elapsed = pd.DataFrame(), 0.0
    else:
        top_accounts = (
            daily_df.groupby("account_id")["outflow"].sum()
                    .nlargest(500).index
        )
        ml_input = daily_df[daily_df["account_id"].isin(top_accounts)]
        ml_df, ml_elapsed = ml_liquidity_forecast(ml_input)

    # Guard: if no ML models trained, fall back to heuristic only
    if ml_df.empty:
        print("  No ML models trained — falling back to heuristic only")
        if heuristic_df.empty:
            print("  ⚠️  Heuristic is ALSO empty — nothing to save. "
                  "Lower MIN_TRANSACTIONS at the top of this file and re-run.")
            liquidity_df = heuristic_df.copy()
            liquidity_df["alert_level"] = pd.Series(dtype=str)
        else:
            liquidity_df = flag_high_risk(heuristic_df)
            liquidity_df = liquidity_df.sort_values(
                "gap_risk_score", ascending=False
            ).reset_index(drop=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        liquidity_df.to_csv(f"{OUTPUT_DIR}/liquidity_output.csv", index=False)
        print(f"\n  Saved → {OUTPUT_DIR}/liquidity_output.csv  ({len(liquidity_df)} rows)")
        benchmark = {
            "backend": "GPU (cuML)" if GPU_ML else "CPU (sklearn)",
            "heuristic_time_sec": round(h_elapsed, 4),
            "ml_training_time_sec": 0,
            "entities_scored": len(liquidity_df),
        }
        with open(f"{OUTPUT_DIR}/benchmark_step3.json", "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"  Saved → {OUTPUT_DIR}/benchmark_step3.json")
        return liquidity_df, benchmark

    # Merge: use ML where available, fall back to heuristic
    h_account = heuristic_df[heuristic_df["entity_type"] == "account"].copy()
    ml_augmented = ml_df.merge(
        h_account[["entity_id", "gap_risk_score"]].rename(
            columns={"gap_risk_score": "heuristic_score"}
        ),
        on="entity_id", how="left"
    )

    # Blend: weight ML 60 %, heuristic 40 %
    ml_augmented["gap_risk_score"] = (
        0.6 * ml_augmented["gap_risk_score"] +
        0.4 * ml_augmented["heuristic_score"].fillna(ml_augmented["gap_risk_score"])
    ).clip(0, 1).round(4)
    ml_augmented = ml_augmented.drop(columns=["heuristic_score"], errors="ignore")

    # Combine ML-scored accounts with remaining heuristic-only accounts
    ml_ids   = set(ml_augmented["entity_id"])
    leftover = heuristic_df[~heuristic_df["entity_id"].isin(ml_ids)]
    liquidity_df = pd.concat([ml_augmented, leftover], ignore_index=True)
    liquidity_df = flag_high_risk(liquidity_df)
    liquidity_df = liquidity_df.sort_values(
        "gap_risk_score", ascending=False
    ).reset_index(drop=True)

    high_risk_count = len(liquidity_df[liquidity_df["gap_risk_score"] >= 0.6])
    print(f"  High-risk entities (score >= 0.6): {high_risk_count:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    liquidity_df.to_csv(f"{OUTPUT_DIR}/liquidity_output.csv", index=False)
    print(f"\n  Saved → {OUTPUT_DIR}/liquidity_output.csv  ({len(liquidity_df)} rows)")

    benchmark = {
        "backend":              "GPU (cuML)" if GPU_ML else "CPU (sklearn)",
        "heuristic_time_sec":   round(h_elapsed,  4),
        "ml_training_time_sec": round(ml_elapsed, 4),
        "entities_scored":      len(liquidity_df),
    }
    with open(f"{OUTPUT_DIR}/benchmark_step3.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"  Saved → {OUTPUT_DIR}/benchmark_step3.json")

    return liquidity_df, benchmark


if __name__ == "__main__":
    liquidity_df, benchmark = run_liquidity_forecast()

    print(f"\n--- Top 15 highest liquidity risk entities ---")
    cols = ["entity_id", "entity_type", "model",
            "pred_inflow", "pred_outflow", "gap_risk_score", "alert_level"]
    display_cols = [c for c in cols if c in liquidity_df.columns]
    if not liquidity_df.empty:
        print(liquidity_df[display_cols].head(15).to_string(index=False))
    else:
        print("  (empty — see warnings above)")

    print(f"\n⚡ Benchmark: {benchmark}")