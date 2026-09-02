#!/usr/bin/env python3
"""Generate the SAGE architecture diagram used in README.md.

The framework figure shipped with the manuscript carries section pointers from
an earlier draft's structure (it cites Stabilize as section III and a section XI
that the final paper does not contain). This script draws the same diagram with
the *final* paper's numbering and pillar content, so the README does not
contradict the article it documents. The published figure remains in
`paper/public/figures/sage_framework.pdf` as the version of record.

Writes a light and a dark variant; README selects between them with <picture>.

Usage:
    python scripts/make_architecture_figure.py
"""

from __future__ import annotations

import pathlib

# Type hints use builtin generics: `from __future__ import annotations` is set
# above and the package targets Python 3.10+.

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets"

# ── Content: mapped to the FINAL paper (sections IV-VII, supplements A-E) ─────

PILLARS: list[dict[str, object]] = [
    {
        "letter": "S",
        "name": "Stabilize",
        "tag": "Closed-loop",
        "ref": "§IV · Supp. A, B.1",
        "heading": "Stability model",
        "bullets": [
            "Closed-loop model (Eq. 1)",
            "Observation fidelity",
            "Progress monotonicity",
            "Bounded context noise",
            "4 monitors: drift, oscillation,",
            "fidelity, convergence",
        ],
        "key": "stabilize",
    },
    {
        "letter": "A",
        "name": "Assess",
        "tag": "Cost-aware",
        "ref": "§V · Supp. C, D",
        "heading": "Autonomy + cost",
        "bullets": [
            "CNSR — completions per $",
            "4 behavioural autonomy criteria",
            "Capability levels L0–L6",
            "Autonomy levels 1–5 + oversight",
            "Cost–success trade-off,",
            "rank inversion",
        ],
        "key": "assess",
    },
    {
        "letter": "G",
        "name": "Govern",
        "tag": "Failures + risk",
        "ref": "§VI · Supp. D, E",
        "heading": "Risk governance",
        "bullets": [
            "10-class failure taxonomy",
            "STRIDE for MCP and A2A",
            "11 threat vectors, 4 critical",
            "5 security boundaries",
            "Each class → a monitor",
            "and a mitigation",
        ],
        "key": "govern",
    },
    {
        "letter": "E",
        "name": "Enforce",
        "tag": "Adaptive",
        "ref": "§VII · Supp. B",
        "heading": "Bounded intervention",
        "bullets": [
            "Adaptive Stability Controller",
            "5 bounded interventions",
            "4 control policies",
            "Cooldown + per-task budget M",
            "Evaluated in §VIII",
            "(E4 ablation, E5 monitor)",
        ],
        "key": "enforce",
    },
]

INPUT_LINES = ["LLM-based", "autonomous agent", "under deployment", "conditions"]
OUTPUT_LINES = [
    "Deployment-oriented",
    "assessment:",
    "stability · cost ·",
    "risk · control",
]

BANNER = "SAGE deployment loop:   Stabilize  →  Assess  →  Govern  →  Enforce"
INSTRUMENTATION = (
    "Shared instrumentation layer — the monitor signals that diagnose "
    "instability drive the controller; the cost model that ranks "
    "configurations prices its interventions"
)
FEEDBACK = "closed-loop feedback:  monitor signals  →  bounded interventions"

# ── Themes ───────────────────────────────────────────────────────────────────

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#ffffff",
        "panel": "#f6f8fa",
        "card": "#ffffff",
        "border": "#d0d7de",
        "text": "#1f2328",
        "muted": "#656d76",
        "arrow": "#57606a",
        "stabilize": "#4f46e5",
        "assess": "#059669",
        "govern": "#d97706",
        "enforce": "#dc2626",
        "stabilize_bg": "#eef2ff",
        "assess_bg": "#ecfdf5",
        "govern_bg": "#fffbeb",
        "enforce_bg": "#fef2f2",
    },
    "dark": {
        "bg": "#0d1117",
        "panel": "#161b22",
        "card": "#161b22",
        "border": "#30363d",
        "text": "#e6edf3",
        "muted": "#9198a1",
        "arrow": "#8b949e",
        "stabilize": "#a5b4fc",
        "assess": "#6ee7b7",
        "govern": "#fcd34d",
        "enforce": "#fca5a5",
        "stabilize_bg": "#1e1b4b",
        "assess_bg": "#064e3b",
        "govern_bg": "#451a03",
        "enforce_bg": "#450a0a",
    },
}

# ── Geometry ─────────────────────────────────────────────────────────────────

W, H = 1424, 524
IO_W, IO_H = 148, 196
CARD_W, CARD_GAP = 238, 26
CARD_X0 = 202
CARD_Y, HEAD_H, CARD_H = 104, 58, 268
BAR_Y, BAR_H = 392, 44
FB_Y = 470
FONT = (
    "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',"
    "Helvetica,Arial,sans-serif"
)


def esc(s: str) -> str:
    """Escape the XML metacharacters that appear in this diagram's text."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def card_x(i: int) -> int:
    """Left edge of the i-th pillar card."""
    return CARD_X0 + i * (CARD_W + CARD_GAP)


def arrow(x1: int, y: int, x2: int, c: str) -> str:
    """Horizontal arrow with an explicit head (no <marker> dependency)."""
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" stroke="{c}" '
        f'stroke-width="2.2" stroke-linecap="round"/>'
        f'<path d="M{x2},{y} L{x2 - 9},{y - 5.5} L{x2 - 9},{y + 5.5} z" fill="{c}"/>'
    )


def arrow_up(x: int, y: int, c: str) -> str:
    """Upward-pointing arrowhead, used where the feedback path re-enters."""
    return f'<path d="M{x},{y} L{x - 5.5},{y + 9} L{x + 5.5},{y + 9} z" fill="{c}"/>'


def build(theme: str) -> str:
    """Render the diagram for one colour theme."""
    t = THEMES[theme]
    o: list[str] = []
    a = o.append

    a(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="{FONT}" role="img" '
        f'aria-label="SAGE framework architecture: four pillars with a feedback path">'
    )
    a(f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>')

    # Banner
    a(
        f'<rect x="{(W - 780) // 2}" y="20" width="780" height="42" rx="8" '
        f'fill="{t["panel"]}" stroke="{t["border"]}"/>'
    )
    a(
        f'<text x="{W // 2}" y="47" text-anchor="middle" font-size="19" '
        f'font-weight="700" fill="{t["text"]}">{esc(BANNER)}</text>'
    )

    # Input / output blocks
    io_y = CARD_Y + 44
    for x, lines, label in (
        (20, INPUT_LINES, "INPUT"),
        (W - 20 - IO_W, OUTPUT_LINES, "OUTPUT"),
    ):
        a(
            f'<rect x="{x}" y="{io_y}" width="{IO_W}" height="{IO_H}" rx="10" '
            f'fill="{t["panel"]}" stroke="{t["border"]}" stroke-width="1.5"/>'
        )
        a(
            f'<text x="{x + IO_W // 2}" y="{io_y + 34}" text-anchor="middle" '
            f'font-size="15" font-weight="700" fill="{t["text"]}">{label}</text>'
        )
        for j, ln in enumerate(lines):
            a(
                f'<text x="{x + IO_W // 2}" y="{io_y + 66 + j * 21}" '
                f'text-anchor="middle" font-size="12.5" '
                f'fill="{t["muted"]}">{esc(ln)}</text>'
            )

    mid = io_y + IO_H // 2
    a(arrow(20 + IO_W + 6, mid, CARD_X0, t["arrow"]))
    a(arrow(card_x(3) + CARD_W + 6, mid, W - 20 - IO_W, t["arrow"]))

    # Pillar cards
    for i, p in enumerate(PILLARS):
        x = card_x(i)
        accent, tint = t[p["key"]], t[f'{p["key"]}_bg']

        a(
            f'<rect x="{x}" y="{CARD_Y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
            f'fill="{t["card"]}" stroke="{accent}" stroke-width="1.8"/>'
        )
        a(
            f'<path d="M{x},{CARD_Y + 10} a10,10 0 0 1 10,-10 h{CARD_W - 20} '
            f'a10,10 0 0 1 10,10 v{HEAD_H - 10} h-{CARD_W} z" fill="{tint}"/>'
        )
        a(
            f'<line x1="{x}" y1="{CARD_Y + HEAD_H}" x2="{x + CARD_W}" '
            f'y2="{CARD_Y + HEAD_H}" stroke="{accent}" stroke-width="1.8"/>'
        )

        a(f'<circle cx="{x + 30}" cy="{CARD_Y + 29}" r="15" fill="{accent}"/>')
        a(
            f'<text x="{x + 30}" y="{CARD_Y + 35}" text-anchor="middle" font-size="16" '
            f'font-weight="700" fill="{t["card"]}">{p["letter"]}</text>'
        )
        a(
            f'<text x="{x + 56}" y="{CARD_Y + 27}" font-size="20" font-weight="700" '
            f'fill="{t["text"]}">{p["name"]}</text>'
        )
        a(
            f'<text x="{x + 56}" y="{CARD_Y + 46}" font-size="12" font-style="italic" '
            f'fill="{t["muted"]}">{esc(str(p["tag"]))}</text>'
        )

        a(
            f'<text x="{x + 16}" y="{CARD_Y + HEAD_H + 26}" font-size="13.5" '
            f'font-weight="700" fill="{t["text"]}">{esc(str(p["heading"]))}</text>'
        )
        for j, b in enumerate(p["bullets"]):  # type: ignore[arg-type]
            yy = CARD_Y + HEAD_H + 50 + j * 20
            if not b.startswith(("fidelity", "rank", "and a", "(E4")):
                a(f'<circle cx="{x + 21}" cy="{yy - 4}" r="2.4" fill="{accent}"/>')
            a(
                f'<text x="{x + 31}" y="{yy}" font-size="12.5" '
                f'fill="{t["text"]}">{esc(b)}</text>'
            )

        a(
            f'<text x="{x + 16}" y="{CARD_Y + CARD_H - 14}" font-size="12" '
            f'font-style="italic" font-weight="600" '
            f'fill="{accent}">{esc(str(p["ref"]))}</text>'
        )

        if i < 3:
            a(arrow(x + CARD_W + 2, mid, x + CARD_W + CARD_GAP - 2, t["arrow"]))

    # Feedback path: Enforce -> Stabilize
    ex, sx = card_x(3) + CARD_W // 2, CARD_X0 + CARD_W // 2
    a(
        f'<path d="M{ex},{CARD_Y + CARD_H} V{FB_Y} H{sx} V{CARD_Y + CARD_H + 11}" '
        f'fill="none" stroke="{t["muted"]}" stroke-width="2.2" stroke-dasharray="7 5"/>'
    )
    a(arrow_up(sx, CARD_Y + CARD_H + 2, t["muted"]))

    # Shared instrumentation layer
    bar_x, bar_w = CARD_X0, card_x(3) + CARD_W - CARD_X0
    a(
        f'<rect x="{bar_x}" y="{BAR_Y}" width="{bar_w}" height="{BAR_H}" rx="8" '
        f'fill="{t["panel"]}" stroke="{t["border"]}" stroke-dasharray="5 3"/>'
    )
    a(
        f'<text x="{bar_x + bar_w // 2}" y="{BAR_Y + 28}" text-anchor="middle" '
        f'font-size="12.5" fill="{t["muted"]}">{esc(INSTRUMENTATION)}</text>'
    )

    a(
        f'<rect x="{W // 2 - 232}" y="{FB_Y - 15}" width="464" height="30" rx="6" '
        f'fill="{t["bg"]}"/>'
    )
    a(
        f'<text x="{W // 2}" y="{FB_Y + 5}" text-anchor="middle" font-size="13" '
        f'font-style="italic" fill="{t["muted"]}">{esc(FEEDBACK)}</text>'
    )

    a("</svg>")
    return "\n".join(o)


def main() -> int:
    """Write both theme variants into docs/assets/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "_dark"
        path = OUT_DIR / f"sage_architecture{suffix}.svg"
        path.write_text(build(theme))
        rel = path.relative_to(OUT_DIR.parent.parent)
        print(f"wrote {rel} ({path.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
