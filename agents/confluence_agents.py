"""
CONFLUENCE — Multi-Agent Decision Layer  (v2 — merged best-of-both)
=====================================================================
4 agents in sequence. Reads REAL pipeline outputs from the cuDF/cuGraph
pipeline; silently falls back to mock data if files are not found, so the
demo never breaks regardless of environment.

Gemini is optional and per-case graceful — if any call fails (rate limit,
network, quota), that case gets a deterministic template brief and the
pipeline continues uninterrupted.

Agents:
  1. GraphInsightAgent      — Translates graph metrics into structured findings
  2. AMLPatternAgent        — Assigns risk category + evidence bullets
  3. LiquidityForecastAgent — Scores liquidity alerts with severity + explanation
  4. CaseCoordinatorAgent   — Merges both signals, ranks, writes briefs

Setup:
  export GEMINI_API_KEY=your_key_here   (or set in Colab secrets)
  Set USE_GEMINI = True below to enable natural-language briefs.

Input:   outputs/clusters_output.csv
         outputs/liquidity_output.csv
Output:  outputs/agent_cases.json   ← dashboard-compatible schema
"""

import ast
import json
import os
import random
import uuid
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

random.seed(42)

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR   = "/content/outputs"          # change to local path if running locally
USE_GEMINI   = False                       # flip to True once GEMINI_API_KEY is set
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ─── Data Loading (real pipeline output → mock fallback) ─────────────────────

def _safe_parse_ids(val):
    """Parse account_ids whether stored as a Python list, JSON, or plain string."""
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [str(val)]


def _load_real_clusters() -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, "clusters_output.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise column names to what the agents expect
    df = df.rename(columns={
        "structural_risk_score": "risk_score",
        "total_amount":          "total_amount_involved",
    })
    # Parse account_ids — may be stored as a stringified list in CSV
    if "account_ids" in df.columns:
        df["account_ids"] = df["account_ids"].apply(_safe_parse_ids)
    else:
        # Reconstruct dummy account list from n_accounts count
        n_col = df["n_accounts"] if "n_accounts" in df.columns else pd.Series([4] * len(df))
        df["account_ids"] = n_col.apply(
            lambda n: [f"ACC-{uuid.uuid4().hex[:6].upper()}" for _ in range(int(n))]
        )
    return df


def _load_real_liquidity() -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, "liquidity_output.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        "pred_inflow":  "predicted_inflow_next_7d",
        "pred_outflow": "predicted_outflow_next_7d",
    })
    return df


# ─── Mock Data (used when real pipeline outputs are not found) ────────────────

PATTERN_TYPES = ["circular_flow", "fan_in_fan_out", "fast_pass_through"]


def _mock_clusters(n: int = 8) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        pattern = random.choice(PATTERN_TYPES)
        n_accounts = random.randint(3, 7)
        if pattern == "fast_pass_through":
            risk = round(random.uniform(0.55, 0.98), 2)
            tw   = round(random.uniform(0.25, 4),    2)
        elif pattern == "circular_flow":
            risk = round(random.uniform(0.4,  0.95), 2)
            tw   = round(random.uniform(1,    48),   2)
        else:
            risk = round(random.uniform(0.3,  0.9),  2)
            tw   = round(random.uniform(2,    72),   2)
        rows.append({
            "cluster_id":           f"MOCK-{uuid.uuid4().hex[:8]}",
            "account_ids":          [f"ACC-{random.randint(10000,99999)}" for _ in range(n_accounts)],
            "pattern_type":         pattern,
            "risk_score":           risk,
            "total_amount_involved":round(random.uniform(2_000, 500_000), 2),
            "time_window_hours":    tw,
        })
    return pd.DataFrame(rows)


def _mock_liquidity(n: int = 8) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        et = random.choice(["account", "branch"])
        eid = f"BR-{random.randint(100,999)}" if et == "branch" else f"ACC-{random.randint(10000,99999)}"
        inflow = round(random.uniform(50_000, 2_000_000), 2)
        outflow = round(inflow * (random.uniform(1.05, 1.6) if random.random() < 0.5
                                  else random.uniform(0.5, 0.98)), 2)
        gap_ratio = max(0.0, (outflow - inflow) / max(inflow, 1))
        rows.append({
            "entity_id":                 eid,
            "entity_type":               et,
            "predicted_inflow_next_7d":  inflow,
            "predicted_outflow_next_7d": outflow,
            "gap_risk_score":            round(min(1.0, gap_ratio * 1.3 + random.uniform(0, 0.1)), 2),
        })
    return pd.DataFrame(rows)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load real data; fall back to mock if files not found."""
    clusters_df  = _load_real_clusters()
    liquidity_df = _load_real_liquidity()

    if clusters_df.empty:
        print("  ⚠️  clusters_output.csv not found — using mock cluster data")
        clusters_df = _mock_clusters()
    else:
        print(f"  ✅ Loaded {len(clusters_df)} real clusters")

    if liquidity_df.empty:
        print("  ⚠️  liquidity_output.csv not found — using mock liquidity data")
        liquidity_df = _mock_liquidity()
    else:
        # Take top 20 riskiest to keep output focused
        liquidity_df = liquidity_df.nlargest(20, "gap_risk_score")
        print(f"  ✅ Loaded {len(liquidity_df)} liquidity entities (top 20 by risk)")

    return clusters_df, liquidity_df


# ─── Severity helpers ─────────────────────────────────────────────────────────

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def _aml_severity(score: float) -> str:
    if score >= 0.85: return "Critical"
    if score >= 0.65: return "High"
    if score >= 0.45: return "Medium"
    return "Low"


def _liq_severity(score: float) -> str:
    if score >= 0.75: return "Critical"
    if score >= 0.55: return "High"
    if score >= 0.35: return "Medium"
    return "Low"


def _format_hours(hours: float) -> str:
    if hours < 1:  return f"{int(hours * 60)} minutes"
    if hours < 48: return f"{hours:.1f} hours"
    return f"{hours / 24:.1f} days"


# ─── Agent 1 — Graph Insight ──────────────────────────────────────────────────

class GraphInsightAgent:
    """
    Translates raw structural graph metrics into clear, structured findings.

    In production: deployed as an independent scalable service consuming
    cuGraph output directly from the pipeline's message bus.
    """

    PATTERN_DESCRIPTIONS = {
        "circular_flow":      "passing money in a closed loop",
        "fan_in_fan_out":     "consolidating funds from many sources then dispersing them",
        "fast_pass_through":  "moving money through with minimal dwell time",
        "circular_flow_scc":  "forming a strongly connected component — tight layering ring",
    }

    def run(self, clusters_df: pd.DataFrame) -> List[Dict[str, Any]]:
        findings = []
        for _, row in clusters_df.iterrows():
            accounts = _safe_parse_ids(row["account_ids"]) if "account_ids" in row.index else []
            # Use n_accounts from row if available — more reliable than len(accounts)
            n = int(row["n_accounts"]) if "n_accounts" in row.index else len(accounts)
            desc = self.PATTERN_DESCRIPTIONS.get(
                row["pattern_type"], "an unusual transaction pattern"
            )
            findings.append({
                "cluster_id":           row["cluster_id"],
                "account_ids":          accounts,
                "n_accounts":           n,
                "pattern_type":         row["pattern_type"],
                "risk_score":           float(row["risk_score"]),
                "total_amount_involved":float(row["total_amount_involved"]),
                "time_window_hours":    float(row.get("time_window_hours", 24)),
                "description": (
                    f"{n} accounts {desc} within "
                    f"{_format_hours(float(row.get('time_window_hours', 24)))}, "
                    f"moving ${float(row['total_amount_involved']):,.0f} total."
                ),
            })
        return findings


# ─── Agent 2 — AML Pattern ────────────────────────────────────────────────────

class AMLPatternAgent:
    """
    Decides which findings are genuinely suspicious, assigns a risk category,
    and generates structured evidence bullets explaining why.

    In production: has access to historical SAR outcomes to recalibrate
    its threshold and category boundaries continuously.
    """

    FLAG_THRESHOLD = 0.35

    def run(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._build_case(f) for f in findings
                if f["risk_score"] >= self.FLAG_THRESHOLD]

    def _build_case(self, f: Dict[str, Any]) -> Dict[str, Any]:
        severity = _aml_severity(f["risk_score"])
        evidence = self._build_evidence(f)
        action_map = {
            "Critical": "Escalate immediately — freeze accounts, file SAR within 24h",
            "High":     "Open formal SAR investigation within 48h",
            "Medium":   "Assign to investigator queue for review this week",
            "Low":      "Log for monitoring; no immediate action required",
        }
        return {
            "cluster_id":           f["cluster_id"],
            "n_accounts":           f["n_accounts"],
            "pattern_type":         f["pattern_type"],
            "risk_score":           f["risk_score"],
            "severity":             severity,
            "total_amount_involved":f["total_amount_involved"],
            "time_window_hours":    f["time_window_hours"],
            "evidence":             evidence,
            "description":          f["description"],
            "recommended_action":   action_map[severity],
            "freeze_recommended":   severity in ("Critical", "High"),
        }

    def _build_evidence(self, f: Dict[str, Any]) -> List[str]:
        bullets = [
            f"Structural risk score {f['risk_score']:.2f} for a "
            f"{f['pattern_type'].replace('_', ' ')} pattern.",
            f"{f['n_accounts']} accounts moved ${f['total_amount_involved']:,.0f} "
            f"in {_format_hours(f['time_window_hours'])}.",
        ]
        if f["time_window_hours"] < 6:
            velocity = f["total_amount_involved"] / max(f["time_window_hours"], 0.1)
            bullets.append(
                f"High velocity: ~${velocity:,.0f}/hour — consistent with deliberate "
                f"rapid layering, not normal account activity."
            )
        if f["n_accounts"] >= 5:
            bullets.append(
                f"{f['n_accounts']}-node ring detected — complexity above typical "
                f"two-party structuring; indicative of organised mule network."
            )
        return bullets


# ─── Agent 3 — Liquidity Forecast ────────────────────────────────────────────

class LiquidityForecastAgent:
    """
    Scores accounts/branches with a meaningful cash-flow gap, assigns severity,
    and writes a plain-language warning a treasury analyst can act on immediately.

    In production: runs on a schedule against the treasury team's rolling
    7-day forecast refreshes.
    """

    ALERT_THRESHOLD = 0.3

    def run(self, liquidity_df: pd.DataFrame) -> List[Dict[str, Any]]:
        alerts = []
        for _, row in liquidity_df.iterrows():
            if row["gap_risk_score"] < self.ALERT_THRESHOLD:
                continue
            alerts.append(self._build_alert(row))
        return alerts

    def _build_alert(self, row: pd.Series) -> Dict[str, Any]:
        inflow  = float(row["predicted_inflow_next_7d"])
        outflow = float(row["predicted_outflow_next_7d"])
        gap     = outflow - inflow
        severity = _liq_severity(float(row["gap_risk_score"]))
        label   = row["entity_type"].capitalize()
        action_map = {
            "Critical": "Immediate treasury review — arrange contingency funding line",
            "High":     "Escalate to treasury desk; funding plan required within 48h",
            "Medium":   "Add to weekly liquidity watchlist",
            "Low":      "Continue routine monitoring",
        }
        explanation = (
            f"{label} {row['entity_id']} is forecast to pay out "
            f"${gap:,.0f} more than it takes in over the next 7 days "
            f"(inflow ${inflow:,.0f} vs outflow ${outflow:,.0f})."
        ) if gap > 0 else (
            f"{label} {row['entity_id']} shows a narrowing cash buffer "
            f"over the next 7 days — inflow still covers outflow but the margin is tightening."
        )
        return {
            "entity_id":                 row["entity_id"],
            "entity_type":               row["entity_type"],
            "severity":                  severity,
            "risk_score":                float(row["gap_risk_score"]),
            "gap_amount":                round(gap, 2),
            "predicted_inflow_next_7d":  round(inflow,  2),
            "predicted_outflow_next_7d": round(outflow, 2),
            "explanation":               explanation,
            "recommended_action":        action_map[severity],
        }


# ─── Gemini Brief Writer ──────────────────────────────────────────────────────

class GeminiBriefWriter:
    """
    Calls the Gemini REST API to generate a natural-language case brief.
    Fails soft on ANY error — the caller always gets None and falls back
    to the deterministic template. A demo should never break because of
    a rate limit or flaky connection.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY,
                 model: str = GEMINI_MODEL, timeout: float = 10.0):
        self.api_key  = api_key
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{model}:generateContent"
        )
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def write_brief(self, prompt: str) -> Optional[str]:
        if not self.available():
            return None
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200},
        }
        try:
            resp = requests.post(
                self.endpoint,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return None


# ─── Agent 4 — Case Coordinator ──────────────────────────────────────────────

class CaseCoordinatorAgent:
    """
    Merges AML cases + liquidity alerts into a single prioritised queue.
    Ranks first by severity tier (Critical > High > Medium > Low), then
    by raw risk score within each tier.

    Writes a plain-language brief for each case — tries Gemini first if
    enabled, falls back to a deterministic template per-case on any failure.

    In production: final aggregation step before cases are pushed to the
    investigator/analyst UI or ticketing system (ServiceNow, Jira, etc.).
    """

    def __init__(self, use_gemini: bool = USE_GEMINI):
        self.gemini = GeminiBriefWriter() if use_gemini else None
        if use_gemini and not (self.gemini and self.gemini.available()):
            print("  ⚠️  USE_GEMINI=True but GEMINI_API_KEY not set — using templates")

    def run(self, aml_cases: List[Dict], liquidity_alerts: List[Dict]) -> List[Dict]:
        unified = []

        for case in aml_cases:
            unified.append({"_type": "AML",       "_score": case["risk_score"],
                             "_severity": case["severity"], "_raw": case})
        for alert in liquidity_alerts:
            unified.append({"_type": "Liquidity",  "_score": alert["risk_score"],
                             "_severity": alert["severity"], "_raw": alert})

        unified.sort(
            key=lambda c: (SEVERITY_RANK.get(c["_severity"], 0), c["_score"]),
            reverse=True,
        )

        queue = []
        for rank, item in enumerate(unified, start=1):
            brief, action = self._write_brief(item)
            queue.append(self._build_output(rank, item, brief, action))

        return queue

    # ── Brief writing ─────────────────────────────────────────────────────────

    def _write_brief(self, item: Dict) -> Tuple[str, str]:
        if self.gemini and self.gemini.available():
            prompt = self._build_prompt(item)
            raw    = self.gemini.write_brief(prompt)
            if raw:
                parsed = self._parse_gemini(raw)
                if parsed:
                    return parsed
        return self._template_brief(item)

    @staticmethod
    def _build_prompt(item: Dict) -> str:
        raw  = item["_raw"]
        role = "AML compliance investigator" if item["_type"] == "AML" \
               else "treasury/liquidity analyst"
        if item["_type"] == "AML":
            facts = (
                f"Pattern: {raw['pattern_type']}  |  "
                f"Accounts: {raw['n_accounts']}  |  "
                f"Amount: ${raw['total_amount_involved']:,.0f}  |  "
                f"Time window: {raw['time_window_hours']:.1f}h  |  "
                f"Severity: {raw['severity']}"
            )
        else:
            facts = (
                f"Entity: {raw['entity_type']} {raw['entity_id']}  |  "
                f"7d inflow: ${raw['predicted_inflow_next_7d']:,.0f}  |  "
                f"7d outflow: ${raw['predicted_outflow_next_7d']:,.0f}  |  "
                f"Gap: ${raw['gap_amount']:,.0f}  |  "
                f"Severity: {raw['severity']}"
            )
        return (
            f"You are writing a case summary for a {role} in an automated "
            f"fraud/liquidity monitoring system called Confluence.\n\n"
            f"Facts:\n{facts}\n\n"
            f"Reply with EXACTLY two lines:\n"
            f"BRIEF: <one sentence, readable in under 10 seconds>\n"
            f"ACTION: <one short concrete next action>"
        )

    @staticmethod
    def _parse_gemini(text: str) -> Optional[Tuple[str, str]]:
        brief, action = None, None
        for line in text.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("BRIEF:"):
                brief = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ACTION:"):
                action = line.split(":", 1)[1].strip()
        return (brief, action) if brief and action else None

    @staticmethod
    def _template_brief(item: Dict) -> Tuple[str, str]:
        raw      = item["_raw"]
        severity = item["_severity"]
        if item["_type"] == "AML":
            brief = (
                f"[{severity}] {raw['pattern_type'].replace('_',' ').title()} — "
                f"{raw['n_accounts']} accounts, ${raw['total_amount_involved']:,.0f} moved "
                f"in {_format_hours(raw['time_window_hours'])}. {raw['description']}"
            )
            action = raw["recommended_action"]
        else:
            brief = (
                f"[{severity}] {raw['entity_type'].capitalize()} {raw['entity_id']} "
                f"liquidity gap risk. {raw['explanation']}"
            )
            action = raw["recommended_action"]
        return brief, action

    # ── Output builder (dashboard-compatible schema) ───────────────────────────

    @staticmethod
    def _build_output(rank: int, item: Dict, brief: str, action: str) -> Dict:
        raw      = item["_raw"]
        is_aml   = item["_type"] == "AML"

        # time-to-act based on severity
        urgency_map = {"Critical": 4, "High": 24, "Medium": 72, "Low": 168}
        time_to_act = urgency_map.get(item["_severity"], 72)

        return {
            # ── Top-level keys the dashboard reads ──
            "cluster_id":        raw.get("cluster_id", f"LIQ-{raw.get('entity_id',rank)}"),
            "pattern_type":      raw.get("pattern_type", f"liquidity_{raw.get('entity_type','alert')}"),
            "n_accounts":        raw.get("n_accounts", 1),
            "final_priority":    round(item["_score"], 4),
            "freeze_recommended":raw.get("freeze_recommended", item["_severity"] == "Critical"),
            "case_type":         item["_type"],
            "severity":          item["_severity"],
            "priority_rank":     rank,
            # ── AML assessment block ──
            "aml_assessment": {
                "risk_category":      item["_severity"],
                "risk_score":         item["_score"],
                "recommended_action": action,
                "evidence":           raw.get("evidence", [brief]),
            },
            # ── Coordinator output block ──
            "coordinator_output": {
                "case_summary":      brief,
                "recommended_action":action,
                "time_to_act_hours": time_to_act,
                "priority_rank":     rank,
            },
            # ── Raw data for downstream use ──
            "raw": raw,
        }


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_agent_pipeline() -> List[Dict]:
    print("=" * 65)
    print("CONFLUENCE — Multi-Agent Advisory Layer  (v2)")
    print("=" * 65)

    # Load data
    print("\n[0/4] Loading pipeline outputs ...")
    clusters_df, liquidity_df = load_data()

    # Agent 1
    print("\n[1/4] GraphInsightAgent — building findings ...")
    findings = GraphInsightAgent().run(clusters_df)
    print(f"  → {len(findings)} cluster findings")

    # Agent 2
    print("\n[2/4] AMLPatternAgent — classifying risk ...")
    aml_cases = AMLPatternAgent().run(findings)
    print(f"  → {len(aml_cases)} AML cases above threshold")

    # Agent 3
    print("\n[3/4] LiquidityForecastAgent — scoring alerts ...")
    liquidity_alerts = LiquidityForecastAgent().run(liquidity_df)
    print(f"  → {len(liquidity_alerts)} liquidity alerts above threshold")

    # Agent 4
    gemini_status = "Gemini enabled" if USE_GEMINI else "template mode (Gemini off)"
    print(f"\n[4/4] CaseCoordinatorAgent — ranking + writing briefs ({gemini_status}) ...")
    coordinator = CaseCoordinatorAgent(use_gemini=USE_GEMINI)
    queue = coordinator.run(aml_cases, liquidity_alerts)
    print(f"  → {len(queue)} cases in final ranked queue")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "agent_cases.json")
    with open(out_path, "w") as f:
        json.dump(queue, f, indent=2, default=str)
    print(f"\n✅ Saved {len(queue)} cases → {out_path}")

    # Print top 3
    print("\n--- Top 3 priority cases ---")
    for case in queue[:3]:
        print(
            f"  #{case['priority_rank']}  {case['cluster_id']}"
            f"  |  severity={case['severity']}"
            f"  |  priority={case['final_priority']:.3f}"
            f"  |  freeze={case['freeze_recommended']}"
        )
        print(f"       {case['coordinator_output']['case_summary'][:120]}...")

    return queue


if __name__ == "__main__":
    run_agent_pipeline()
