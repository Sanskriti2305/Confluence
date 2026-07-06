# 🌊 Confluence

### *Where flows meet — real-time money-flow intelligence for banks*

> One live transaction graph. Two critical bank decisions. Built to catch what batch jobs and single-transaction rules can't.

## 🚀 Live Demo

**🌐 Streamlit App:** https://confluence-xn.streamlit.app/

---

## ⚡ TL;DR

Banks watch money move for two reasons: **catching launderers** and **avoiding cash crunches**. Today they do both poorly—overnight batch jobs, single-transaction rules, and slow spreadsheets—when the real answer is hidden in the *shape* of how money flows between accounts.

**Confluence turns every transaction into a graph, scores it in real time using GPU acceleration, and tells the right person—investigator or treasury analyst—exactly what's happening and why, before the money disappears.**

---

## ✅ Brief Alignment (Track 2 — Data Intelligence Tool)

| Brief requirement                                          | How Confluence meets it                                                                        |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Clear real-world user & problem**                        | AML compliance investigators and treasury analysts at financial institutions                   |
| **A specific decision that depends on data**               | *"Is this account cluster a genuine mule ring?"* and *"Is a liquidity gap beginning to form?"* |
| **Pipeline: ingest → clean → analyze → model → visualize** | Cloud Storage → cuDF → BigQuery → cuGraph/cuML → Multi-Agent Layer → Looker                    |
| **Useful output**                                          | Ranked investigation queue, cluster risk scores, liquidity alerts and evidence summaries       |
| **Decision acceleration**                                  | Near-real-time graph analytics instead of overnight batch processing                           |
| **Google Cloud services**                                  | Cloud Storage, BigQuery, Gemini Enterprise Agent Platform, GKE, Looker                         |
| **NVIDIA acceleration**                                    | RAPIDS (cuDF, cuGraph, cuML) running on NVIDIA GPUs                                            |

---

## 🔍 The Problem

### 🕵️ AML Compliance

Investigators often begin each day with hundreds of alerts, the overwhelming majority of which are false positives. Traditional rule-based systems evaluate transactions individually, allowing sophisticated **mule rings** that distribute money across multiple accounts to remain hidden.

### 💰 Treasury Management

Liquidity issues are frequently identified only after cash outflows have already exceeded expectations. Existing systems monitor balances rather than continuously analyzing the flow of funds across the banking network.

**Both problems share the same root cause:** banks analyze isolated transactions instead of understanding the **network structure** created by those transactions.

---

## 💡 Core Insight

Money moving through a banking system naturally forms a **graph**.

* **Accounts → Nodes**
* **Transactions → Edges**

| Pattern             | Graph Perspective                                                            |
| ------------------- | ---------------------------------------------------------------------------- |
| 🕵️ Mule Ring       | Circular transfers, fan-in/fan-out behaviour, rapid pass-through             |
| 💧 Liquidity Stress | Sustained imbalance between inflows and outflows across nodes or communities |

One graph. One analytics engine. Two completely different business decisions.

---

## 👥 Target Users

| User                     | Current Challenge                    | Confluence Solution                                             |
| ------------------------ | ------------------------------------ | --------------------------------------------------------------- |
| 🕵️ AML Investigator     | Large queue of false-positive alerts | Prioritized structural-risk clusters with supporting evidence   |
| 💰 Treasury Analyst      | Detects liquidity issues too late    | Early warnings before liquidity stress becomes critical         |
| 🧑‍💼 Compliance Manager | Manual case prioritization           | Automatically ranked investigations with AI-generated summaries |

---

## 🏗️ System Architecture

```text
Transactions Stream
        │
        ▼
Cloud Storage
        │
        ▼
GPU Data Cleaning (cuDF)
        │
        ▼
BigQuery Warehouse
        │
        ▼
Live Transaction Graph (cuGraph)
        │
   ┌────┴────┐
   ▼         ▼
AML Lens   Liquidity Lens
   └────┬────┘
        ▼
Multi-Agent Advisory Layer
        │
        ▼
Alerts • Case Queue • Looker Dashboard
```

---

## 🤖 Multi-Agent System

Rather than relying on a single AI model, Confluence divides responsibilities across specialized agents.

| Agent                       | Responsibility                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------- |
| 🕸️ Graph Builder           | Maintains the continuously updated transaction graph                                  |
| 🕵️ AML Pattern Agent       | Detects mule-ring structures using graph analytics                                    |
| 💧 Liquidity Forecast Agent | Predicts emerging liquidity stress                                                    |
| 🧑‍⚖️ Case Coordinator      | Combines signals, prioritizes investigations and generates natural-language summaries |

Agents are designed to run on **Gemini Enterprise Agent Platform** and scale independently through **Google Kubernetes Engine (GKE)**.

---

## ⚡ Why GPU Acceleration Matters

Acceleration is not simply about faster execution—it fundamentally changes what decisions are possible.

| Traditional Pipeline                                         | GPU-Accelerated Pipeline                                 |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| Overnight AML batch jobs                                     | Near-real-time detection while funds can still be frozen |
| CPU graph libraries struggle with large transaction networks | cuGraph processes bank-scale graphs efficiently          |
| pandas preprocessing becomes a bottleneck                    | cuDF accelerates the same workflow on GPUs               |
| Forecasts updated infrequently                               | Faster training enables more frequent model updates      |

**Acceleration directly improves decision quality—not just benchmark numbers.**

---

## 🧰 Technology Stack

### Google Cloud

| Service                          | Purpose                       |
| -------------------------------- | ----------------------------- |
| Cloud Storage                    | Raw transaction ingestion     |
| BigQuery                         | Transaction warehouse         |
| Gemini Enterprise Agent Platform | Multi-agent orchestration     |
| Google Kubernetes Engine         | Agent deployment and scaling  |
| Looker                           | Dashboards and investigations |

### NVIDIA

| Technology  | Purpose                                 |
| ----------- | --------------------------------------- |
| RAPIDS      | GPU data science ecosystem              |
| cuDF        | Data processing and feature engineering |
| cuGraph     | Graph analytics and community detection |
| cuML        | Machine learning models                 |
| NVIDIA GPUs | High-performance compute layer          |

---

## 📊 Dataset

Confluence is built using the **PaySim synthetic financial transactions dataset**, a widely used benchmark dataset for fraud detection and AML research.

PaySim simulates realistic mobile money transactions while preserving privacy by avoiding real customer data. It contains millions of synthetic banking transactions—including transfers, cash-outs, payments and merchant activity—making it well suited for experimenting with transaction networks, graph analytics and financial risk detection.

Using a synthetic dataset enables reproducible research while reflecting many behavioural patterns observed in real financial systems.

---

## 📈 Benchmarking Acceleration

| Benchmark                     | Purpose                                 |
| ----------------------------- | --------------------------------------- |
| pandas vs cuDF                | GPU-accelerated preprocessing           |
| NetworkX vs cuGraph           | Large-scale graph analytics             |
| Batch processing vs streaming | Reduction in time-to-detection          |
| CPU vs GPU training           | Faster model retraining for forecasting |

---

## 🌟 Why Confluence

* Combines **AML intelligence** and **liquidity monitoring** into a single graph analytics platform.
* Uses **graph-native analysis** instead of relying solely on transaction-level rules.
* Demonstrates how **GPU acceleration** improves operational decision-making rather than simply improving benchmark scores.
* Integrates **Google Cloud**, **NVIDIA RAPIDS**, **graph analytics**, and a **multi-agent AI architecture** into one end-to-end system.
* Designed around workflows already used by compliance and treasury teams, minimizing adoption friction.
