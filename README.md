# 🌊 Confluence — Real-Time Money-Flow Intelligence

> One live transaction graph. Two critical bank decisions. Built on RAPIDS + Gemini.

## Project Structure

```
confluence/
├── notebooks/
│   ├── 01_data_pipeline.py        ← Data loading, cleaning (cuDF benchmark)
│   ├── 02_graph_analysis.py       ← Graph build + AML detection (cuGraph)
│   ├── 03_liquidity_forecast.py   ← Liquidity risk scoring (cuML)
│   └── 04_benchmark_report.py     ← CPU vs GPU comparison table
├── agents/
│   └── confluence_agents.py       ← 4-agent orchestration layer (Gemini API)
├── dashboard/
│   └── app.py                     ← Streamlit dashboard
├── data/                          ← Place PaySim CSV here
│   └── .gitkeep
├── outputs/                       ← Pipeline writes CSVs here
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Dataset
Download PaySim from Kaggle and place it in `data/`:
- https://www.kaggle.com/datasets/ealaxi/paysim1
- File: `PS_20174392719_1491204439457_log.csv`

## Quick Start (Colab / Vertex AI with GPU)
```python
# 1. Install RAPIDS (in Colab with GPU runtime)
!pip install cudf-cu12 cugraph-cu12 cuml-cu12 --extra-index-url https://pypi.nvidia.com

# 2. Run pipeline
!python notebooks/01_data_pipeline.py
!python notebooks/02_graph_analysis.py
!python notebooks/03_liquidity_forecast.py

# 3. Launch dashboard
!streamlit run dashboard/app.py
```
