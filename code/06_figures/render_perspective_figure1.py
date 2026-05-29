#!/usr/bin/env python3
"""Render comprehensive Perspective Figure 1: saturation-gap mechanism (English)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/analysis/figures/perspective_figure1_saturation_gap_v2.png"
OUT_PDF = OUT.with_suffix(".pdf")

# Corpus stats cited in Perspective.docx
N = 1388
PATH_ALPHA_PCT = 0.58
PATH_BETA_PCT = 99.42
FRAME_C_PCT = 97.77
GOV_PERF_PCT = 67.36
REG_CONTEXT_PCT = 80.3
ETHNIC_LEX_PCT = 20.17
AGENCY_FDA_PCT = 90.1


def _box(ax, xy, w, h, text, fc="#EDF2F7", ec="#2D3748", fs=8, lw=1.2, style="round,pad=0.02"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=style,
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transData,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#1A202C",
        wrap=True,
        zorder=3,
    )
    return patch


def _arrow(ax, p0, p1, color="#4A5568", lw=1.4, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)


def render(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Layer 0: institutional context ─────────────────────────────────────
    _box(
        ax,
        (0.4, 8.55),
        13.2,
        1.15,
        "FDA AI device public summaries (510[k] pathway)  ·  N = 1,388 full texts\n"
        "Primary genre goal: demonstrate safety & effectiveness for regulatory clearance — not host value conflict",
        fc="#EBF8FF",
        ec="#2B6CB0",
        fs=8.5,
    )

    # ── Layer 1: two parallel pressures ────────────────────────────────────
    _box(
        ax,
        (0.5, 6.55),
        4.0,
        1.55,
        "FIELD-LEVEL GAP\n(structured omission)\n\nMissing or thin public fields:\n"
        "demographics · model development ·\nclinical validation (cf. disclosure audits)",
        fc="#FFF5F5",
        ec="#C53030",
        fs=7.5,
    )
    _box(
        ax,
        (5.0, 6.55),
        4.0,
        1.55,
        "TECHNICAL–PROCEDURAL SATURATION\n(auditable fullness)\n\nHigh-frequency lexicon:\n"
        "subgroup · sensitivity · risk management ·\npost-market surveillance · AUC / V&V",
        fc="#F0FFF4",
        ec="#276749",
        fs=7.5,
    )
    _box(
        ax,
        (9.5, 6.55),
        4.0,
        1.55,
        "READER ILLUSION\n(performative transparency)\n\n“More disclosure” feels like\n"
        "more fairness accountability\n(seeing without contesting)",
        fc="#FFFAF0",
        ec="#C05621",
        fs=7.5,
    )

    # ── Layer 2: saturation–gap paradox (center) ───────────────────────────
    _box(
        ax,
        (2.2, 4.85),
        9.6,
        1.35,
        "SATURATION–GAP PARADOX\n"
        "Gaps are not simply empty — they are occupied by auditable technical–procedural grammar\n"
        f"Frame C (performance register): {FRAME_C_PCT:.2f}%  ·  "
        f"Normative anchor (Path α): {PATH_ALPHA_PCT:.2f}% (n = 8)  ·  "
        f"Path β (no normative anchor): {PATH_BETA_PCT:.2f}%\n"
        f"Ethnicity-related terms visible in {ETHNIC_LEX_PCT:.2f}% of files — "
        "yet normative fairness rarely becomes the narrative spine",
        fc="#FAF5FF",
        ec="#6B46C1",
        fs=8,
    )

    # ── Layer 3: three mechanisms ──────────────────────────────────────────
    mech_y = 2.55
    mech_h = 1.85
    mech_w = 4.1
    _box(
        ax,
        (0.45, mech_y),
        mech_w,
        mech_h,
        "Mechanism 1\nSEMANTIC ABSTRACTION\n\nFairness conflicts → subgroup,\n"
        "generalizability, representation\n(ML vs demographic sense; n = 121 w/ term)\n"
        "Diversity visible; structural injustice muted",
        fc="#E6FFFA",
        ec="#319795",
        fs=7.2,
    )
    _box(
        ax,
        (4.95, mech_y),
        mech_w,
        mech_h,
        "Mechanism 2\nMETRICIZATION & PROCEDURAL GOVERNANCE\n\n"
        f"Performance grammar ≈ {FRAME_C_PCT:.2f}%  ·  Governance + performance ≈ {GOV_PERF_PCT:.2f}%\n"
        f"≈ {REG_CONTEXT_PCT:.1f}% of anchor terms in compliance/regulatory context\n"
        "Explanation ≠ justification; auditability crowds out harm & remedy",
        fc="#FEEBC8",
        ec="#DD6B20",
        fs=7.0,
    )
    _box(
        ax,
        (9.45, mech_y),
        mech_w,
        mech_h,
        "Mechanism 3\nACCOUNTABILITY REDISTRIBUTION\n\n"
        f"≈ {AGENCY_FDA_PCT:.1f}% agency attribution → FDA / device / compliance chain\n"
        "Clinicians framed as assisted users, not public accountability subjects\n"
        "Residual risk shifts to post-market use & clinical practice",
        fc="#E9D8FD",
        ec="#805AD5",
        fs=7.0,
    )

    # ── Layer 4: outcome vs reform ─────────────────────────────────────────
    _box(
        ax,
        (0.45, 0.35),
        6.2,
        1.55,
        "GENRE-LEVEL OUTCOME\nDiscursive self-defense\n\nPublic spine answers:\n"
        "“Are metrics & procedures auditable?”\nnot “Who is responsible for distributive fairness?”",
        fc="#FED7D7",
        ec="#9B2C2C",
        fs=7.5,
    )
    _box(
        ax,
        (7.0, 0.35),
        6.55,
        1.55,
        "REFORM DIRECTION\nContestable fairness\n\nShift from checkbox disclosure →\n"
        "mandated normative slots: harm narrative · unassessed populations ·\n"
        "responsibility chain · traceable remedy (disclosable AND contestable)",
        fc="#C6F6D5",
        ec="#22543D",
        fs=7.2,
    )

    # ── Arrows ─────────────────────────────────────────────────────────────
    _arrow(ax, (2.5, 8.55), (2.5, 8.1), color="#2B6CB0")
    _arrow(ax, (7.0, 8.55), (7.0, 8.1), color="#2B6CB0")
    _arrow(ax, (11.5, 8.55), (11.5, 8.1), color="#2B6CB0")

    _arrow(ax, (2.5, 6.55), (5.5, 6.2), color="#C53030")
    _arrow(ax, (7.0, 6.55), (7.0, 6.2), color="#276749")
    _arrow(ax, (11.5, 6.55), (9.0, 6.2), color="#C05621")

    _arrow(ax, (2.5, 6.55), (4.5, 5.5), color="#C53030", rad=0.08)
    _arrow(ax, (7.0, 6.55), (7.0, 6.2), color="#276749")
    _arrow(ax, (11.5, 6.55), (9.5, 5.5), color="#C05621", rad=-0.08)

    _arrow(ax, (7.0, 4.85), (7.0, 4.45), color="#6B46C1", lw=1.8)

    _arrow(ax, (2.5, 4.85), (2.5, 4.45), color="#6B46C1")
    _arrow(ax, (11.0, 4.85), (11.0, 4.45), color="#6B46C1")

    for x in (2.5, 7.0, 11.5):
        _arrow(ax, (x, 2.55), (x, 1.95), color="#4A5568")

    _arrow(ax, (3.5, 0.9), (7.0, 0.9), color="#22543D", lw=2.0, style="-|>", rad=0.15)

    ax.text(
        7.0,
        0.15,
        "Figure 1. Saturation–gap mechanism in FDA AI public summaries: from field omission and technical saturation "
        "to genre-level discursive self-defense, with three structural production mechanisms and a contestable-fairness reform horizon.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#2D3748",
        style="italic",
    )

    fig.suptitle(
        "Saturation–Gap Mechanism in Public Regulatory Summaries",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    render(OUT)
