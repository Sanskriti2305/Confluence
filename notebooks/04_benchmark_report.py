"""
Confluence — Step 4: Benchmark Report
=======================================
Collects the timing JSON files from steps 1-3 and produces a clean
comparison table (CPU vs GPU) for the demo slide.

If you have GPU numbers, manually add them below (or re-run on a GPU
runtime and this script reads them automatically).

Output: ../outputs/benchmark_report.csv
        prints a formatted table to stdout
"""

import json
import os
import pandas as pd

OUTPUT_DIR = "/content/outputs"
FILES = {
    "Step 1 — Data Cleaning":        "benchmark_step1.json",
    "Step 2 — Graph Build":          "benchmark_step2.json",
    "Step 3 — Liquidity ML Training": "benchmark_step3.json",
}

# ── If you ran on CPU AND GPU, paste GPU numbers here ─────────────────────────
# (or just leave None and the script will mark them as pending)
KNOWN_GPU_TIMES = {
    "Step 1 — Data Cleaning":         None,  # e.g. 1.2
    "Step 2 — Graph Build":           None,  # e.g. 0.8
    "Step 3 — Liquidity ML Training": None,  # e.g. 3.1
}
# ─────────────────────────────────────────────────────────────────────────────


def load_benchmark(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_primary_time(data: dict) -> float:
    """Extract the most relevant timing field from a benchmark JSON."""
    for key in [
        "cleaning_time_sec", "graph_build_time_sec",
        "ml_training_time_sec", "total_time_sec"
    ]:
        if key in data and data[key] is not None:
            return data[key]
    return None


def build_report():
    rows = []
    for label, filename in FILES.items():
        path = os.path.join(OUTPUT_DIR, filename)
        data = load_benchmark(path)
        if data is None:
            cpu_time = None
            backend  = "not run yet"
        else:
            cpu_time = get_primary_time(data)
            backend  = data.get("backend", "unknown")

        gpu_time = KNOWN_GPU_TIMES.get(label)

        if cpu_time and gpu_time:
            speedup = round(cpu_time / gpu_time, 1)
            speedup_str = f"{speedup}×"
        else:
            speedup_str = "pending GPU run"

        rows.append({
            "Pipeline Stage":     label,
            "CPU Time (s)":       cpu_time if cpu_time else "—",
            "GPU Time (s)":       gpu_time if gpu_time else "pending",
            "Speedup":            speedup_str,
            "Backend Detected":   backend,
        })

    report = pd.DataFrame(rows)
    return report


def print_report(report: pd.DataFrame):
    print("\n" + "=" * 75)
    print("CONFLUENCE — GPU vs CPU Acceleration Benchmark")
    print("=" * 75)
    print(report.to_string(index=False))
    print("=" * 75)
    print("\n💡 To fill in GPU Times:")
    print("   1. Open a Colab notebook with GPU runtime (T4 or A100)")
    print("   2. Run: !pip install cudf-cu12 cugraph-cu12 cuml-cu12 \\")
    print("           --extra-index-url https://pypi.nvidia.com")
    print("   3. Re-run steps 1-3")
    print("   4. Copy the *_time_sec values from outputs/benchmark_step*.json")
    print("   5. Paste into KNOWN_GPU_TIMES at the top of this script")
    print("\n📊 This table becomes your core benchmark slide.")


if __name__ == "__main__":
    report = build_report()
    print_report(report)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report.to_csv(f"{OUTPUT_DIR}/benchmark_report.csv", index=False)
    print(f"\nSaved → {OUTPUT_DIR}/benchmark_report.csv")
