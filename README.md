# 🌊 Confluence
### *Where flows meet — real-time money-flow intelligence for banks*

> One live transaction graph. Two critical bank decisions. Built to catch what batch jobs and single-transaction rules can't.

---

## ⚡ TL;DR

Banks watch money move for two reasons: **catching launderers** and **avoiding cash crunches**. Today they do both badly — overnight batch jobs, single-transaction rules, and slow spreadsheets, when the real answer is sitting in the *shape* of how money flows between accounts.

**Confluence turns every transaction into a graph, scores it in real time on GPU, and tells the right person — investigator or treasury analyst — exactly what's happening and why, before the money disappears.**

---

## ✅ Brief Alignment (Track 2 — Data Intelligence Tool)

| Brief requirement | How Confluence meets it |
|---|---|
| **Clear real-world user & problem** | AML compliance investigators + treasury analysts at any bank — existing roles, existing mandate, existing budget (see "Who Actually Uses This") |
| **A specific decision that depends on data** | *"Is this account cluster a real mule ring, and how urgently should I escalate it?"* / *"Is a liquidity gap forming here, and do we need to act before it becomes a crunch?"* |
| **Pipeline: ingest → clean → analyze → model → visualize** | Cloud Storage (ingest) → cuDF (clean) → BigQuery (structure) → cuGraph/cuML (analyze & model) → Looker + agent layer (visualize/decide) |
| **Useful output** | Ranked case queue, risk scores per cluster, liquidity early-warning alerts, plain-language evidence brief |
| **Evidence acceleration improves the decision** | Near-real-time graph scoring vs overnight batch = money can still be frozen / crunch can still be averted, not discovered after the fact (see "Why Speed Isn't a Buzzword Here" + benchmark table) |
| **2+ Google Cloud tools used** | Cloud Storage, BigQuery, GKE, Gemini Enterprise Agent Platform, Looker *(5 used)* |
| **2+ NVIDIA acceleration tools used** | NVIDIA RAPIDS (cuDF, cuGraph, cuML), NVIDIA GPUs on Google Cloud *(directly covers 2 of the 4 listed categories)* |

---

## 🔍 The Problem, In Two Scenes

**Scene 1 — The Compliance Floor**
An investigator's queue has 400 flagged transactions this morning. ~90%+ are false positives — a rule tripped because *one* transaction crossed a threshold. Buried somewhere in there is an actual **mule ring**: five accounts quietly passing money in a circle to dodge detection. No single-transaction rule will ever see it — you can only see it by looking at the *network*, not the transaction.

**Scene 2 — The Treasury Desk**
An analyst finds out about a liquidity gap the same way most banks do: after it's already a problem. Outflows quietly outpaced inflows for days. Nobody was watching the *rate*, just the balance. (This exact failure mode is what took down SVB.)

**Both scenes have the same root cause:** the data needed to catch it exists, but nobody's watching the *flow* — only the snapshot.

---

## 💡 The Insight

Money moving between accounts is a **graph**:
- Accounts → nodes
- Transactions → edges

| Pattern | What it looks like in the graph |
|---|---|
| 🕵️ Mule ring | A *structural* pattern — circular flows, fast pass-through, fan-in/fan-out |
| 💧 Liquidity stress | A *flow* pattern — outflow rate outpacing inflow rate at a node or cluster |

**Same graph. Same engine. Two lenses.** That's why this isn't "AML tool + liquidity tool glued together" — it's one live pipeline, split into two questions.

---

## 👥 Who Actually Uses This

| Who | What they're staring at right now | What Confluence gives them |
|---|---|---|
| 🕵️ **AML Investigator** | A queue of mostly-false-positive alerts | A short list of *real* structural risk clusters, ranked, with evidence |
| 💰 **Treasury Analyst** | A balance that already dropped | An early warning *before* the gap forms |
| 🧑‍💼 **Compliance Team Lead** | Case backlog, manual prioritization | Auto-prioritized queue + drafted SAR narrative |

These people already exist, already have budget, already have a mandate. This upgrades a tool they use daily — it doesn't ask anyone to adopt something new.

---

## 🏗️ How It Works

```
Transactions stream in
        │
        ▼
 ☁️  Cloud Storage  →  🧹 cuDF cleaning  →  🗄️ BigQuery warehouse
        │
        ▼
 🕸️  Live Transaction Graph  (cuGraph)
        │
   ┌────┴────┐
   ▼         ▼
🕵️ AML Lens   💧 Liquidity Lens
(mule-ring    (cash-flow gap
 structure)    forecasting)
   └────┬────┘
        ▼
 🤖 Multi-Agent Advisory Layer
        │
        ▼
 📊 Ranked case queue + alerts  →  Looker dashboard
```

---

## 🤖 The Agent Team

Instead of one black-box model, four specialists — because a real investigator asks several different questions, not one:

| Agent | Job |
|---|---|
| 🕸️ **Graph Builder** | Keeps the live transaction graph current as data streams in |
| 🕵️ **AML Pattern Agent** | Hunts for mule-ring structure using cuGraph |
| 💧 **Liquidity Forecast Agent** | Predicts cash-flow gaps before they happen |
| 🧑‍⚖️ **Case Coordinator** | Merges both signals, prioritizes the queue, writes the plain-language brief |

*Runs on Gemini Enterprise Agent Platform, deployed on GKE — each agent scales independently (only the graph-crunching agents need GPU nodes).*

---

## ⚡ Why Speed Isn't a Buzzword Here

This is the part that actually matters to the judges — acceleration isn't "nice," it's the line between two completely different outcomes:

| 🐢 Without acceleration | 🚀 With acceleration |
|---|---|
| AML scoring runs overnight — mule ring found *tomorrow*, money's already gone | Near real-time scoring — flagged *today*, while it can still be frozen |
| CPU graph libraries choke on millions of nodes/edges | cuGraph handles it at bank-scale, fast |
| Feature engineering in pandas slows down as data grows | Same code, `cudf.pandas` — dramatically faster |
| Forecast models retrained weekly (stale) | GPU training makes daily retraining realistic (fresh) |

**One sentence version:** *acceleration is the difference between "we can still freeze this" and "it's already gone."*

---

## 🧰 Stack

**Google Cloud — Data & Application Layer**
| Tool | Role in Confluence |
|---|---|
| `Cloud Storage` | Landing zone for raw transaction data |
| `BigQuery` | Structured warehouse for transactions, case history, outcomes |
| `Gemini Enterprise Agent Platform` | Multi-agent orchestration |
| `Google Kubernetes Engine (GKE)` | Deploys & independently scales each agent |
| `Looker` | Compliance/treasury dashboard, case queue, network visualizations |

**NVIDIA — Acceleration Layer**
| Tool | Role in Confluence |
|---|---|
| `NVIDIA RAPIDS` | Umbrella suite powering cuDF (cleaning), cuGraph (graph analysis), cuML (scoring/forecasting) |
| `cuDF / cudf.pandas` | GPU-accelerated cleaning & feature engineering, same code as pandas |
| `NVIDIA GPUs on Google Cloud` | Underlying compute for all of the above |

---

## 📊 Data

Real bank data can't be used (privacy/regulation) — so this runs on **public synthetic AML transaction-graph datasets** (e.g. IBM AMLSim or similar), scaled to millions of transactions to reflect real bank volume. This is standard practice in AML research, not a shortcut.

---

## 📈 Proving Acceleration (Not Just Claiming It)

| Benchmark | What it shows |
|---|---|
| `pandas` vs `cudf.pandas` | Same code, cleaning millions of transactions |
| `NetworkX` vs `cuGraph` | Community/cycle detection at bank-scale graph size |
| Overnight batch vs near-real-time | Time-to-flag for a mule ring — directly maps to "can we still act?" |
| CPU vs GPU model training *(stretch)* | Supports the case for daily forecast retraining |

---

## 🌟 Why This Stands Out

- **Not another fraud-classification demo** — mule rings are *structurally* invisible without a graph. This uses the right tool, not a forced one.
- **Two real, budgeted problems, one pipeline** — AML and liquidity risk are usually two separate products. Unifying them on one graph is the actual idea.
- **Speed = decision quality, not just a benchmark slide** — the acceleration story has a direct dollar/risk consequence.
- **Zero adoption friction** — the users already work inside this exact kind of tool today.

---

## ✅ Build Checklist

- [ ] Dataset + graph schema finalized
- [ ] cuDF cleaning pipeline
- [ ] cuGraph mule-ring detection
- [ ] Liquidity forecasting model
- [ ] 4 agents wired up on Gemini Enterprise Agent Platform / GKE
- [ ] Looker dashboard
- [ ] Acceleration benchmarks run + documented
- [ ] Final demo walkthrough
