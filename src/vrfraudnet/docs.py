"""Architecture facts as DATA, so the documentation and the diagram are generated from one place and cannot drift from each other.

`vrfraudnet docs` writes docs/figures/pipeline.svg and prints the stage table used in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import html
from pathlib import Path

STAGES = [
    {"id": "0", "name": "Time-conditioned spectral encoder", "module": "models/spectral.py", "input": "x [N,F]; sparse L [N,N] (edges before t); t",
     "output": "z_g [N,64]", "fits_on": "train rows; graph snapshots strictly before the batch's earliest time", "failure": "no graph: bypassed (past-only tabular features)"},
    {"id": "1", "name": "LightGBM triage + threshold gate", "module": "models/triage.py", "input": "[x, z_g]", "output": "p_triage in [0,1]; route in {accept, reject, escalate}",
     "fits_on": "train (early stopping on VAL_SELECT); thresholds on VAL_SELECT", "failure": "band is inclusive: T_low <= p <= T_high escalates"},
    {"id": "R", "name": "Retrieval (inside Stage 2)", "module": "retrieval/index.py", "input": "summary of the review-band case", "output": "<=4 verified TRAIN exemplars",
     "fits_on": "verified training exemplars only", "failure": "fewer or none is valid; never back-filled"},
    {"id": "2", "name": "Schema-constrained rationale model", "module": "models/rationale.py, models/grammar.py", "input": "evidence, p_triage, demonstrations",
     "output": "JSON: verdict, probability, <=8 typed claims, evidence_refs", "fits_on": "LoRA on verified review-band TRAIN targets",
     "failure": "token limit / parse failure => treated as verification failure"},
    {"id": "3", "name": "Fixed deterministic verifier", "module": "verifier/", "input": "rationale + trusted evidence snapshot", "output": "V in {0,1}; claim-level audit",
     "fits_on": "nothing: not trainable", "failure": "unknown evidence is 'unverifiable', never 'absent'; rejection => human review"},
    {"id": "4", "name": "Isotonic mixer + split-conformal", "module": "models/mixer.py, calibration/conformal.py", "input": "p_triage, eligible p_rationale, verifier state",
     "output": "calibrated risk; prediction set", "fits_on": "mixer: VAL_SELECT; conformal: VAL_CALIBRATION (disjoint)", "failure": "rejected rationale probability is never read"},
]


def pipeline_svg() -> str:
    w, bw, bh, gap = 920, 250, 54, 22
    h = 40 + len(STAGES) * (bh + gap)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif" font-size="13">',
           f'<rect width="{w}" height="{h}" fill="#fbfbf8"/>', '<text x="20" y="26" font-weight="600">VR-FraudNet inference path (generated from vrfraudnet.docs.STAGES)</text>']
    y = 44
    for s in STAGES:
        fill = "#e8f1fb" if s["id"] in ("0", "1", "4") else "#fdf1e3" if s["id"] in ("2", "R") else "#efe8f7"
        out.append(f'<rect x="20" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="#555"/>')
        out.append(f'<text x="32" y="{y + 22}" font-weight="600">Stage {html.escape(s["id"])}: {html.escape(s["name"][:30])}</text>')
        out.append(f'<text x="32" y="{y + 42}" fill="#444">{html.escape(s["output"][:38])}</text>')
        out.append(f'<text x="{bw + 40}" y="{y + 22}" fill="#333">in:  {html.escape(s["input"][:75])}</text>')
        out.append(f'<text x="{bw + 40}" y="{y + 42}" fill="#333">fit: {html.escape(s["fits_on"][:75])}</text>')
        if y + bh + gap < h - 20:
            out.append(f'<line x1="{20 + bw // 2}" y1="{y + bh}" x2="{20 + bw // 2}" y2="{y + bh + gap}" stroke="#555" marker-end="url(#a)"/>')
        y += bh + gap
    out.insert(1, '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#555"/></marker></defs>')
    out.append("</svg>")
    return "\n".join(out)


def stage_table_markdown() -> str:
    rows = ["| Stage | Module | Input | Output | Fitted on | Failure policy |", "|---|---|---|---|---|---|"]
    for s in STAGES:
        rows.append(f"| {s['id']} {s['name']} | `{s['module']}` | {s['input']} | {s['output']} | {s['fits_on']} | {s['failure']} |")
    return "\n".join(rows)


def write_figures(docs_dir: Path) -> Path:
    p = Path(docs_dir) / "figures" / "pipeline.svg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(pipeline_svg(), encoding="utf-8")
    return p
