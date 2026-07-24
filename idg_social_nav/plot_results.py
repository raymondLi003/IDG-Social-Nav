"""Figures from the run_eval CSVs.

Figure 1 (per scenario): the goal step cost and social-cost Pareto view.
The fixed-blend baseline is used to see which advise-planner blend gets the best combo
We compare other validators (always_obey, oracle, SAC/PPO, and LLMs)
to the Pareto front of the fixed-blend sweep.

Figure 2: grouped bars of good/bad disobedience rates and override
precision/recall per validator, one panel per scenario.

Usage:
    python -m idg_social_nav.plot_results          # newest CSV in eval_results/
    python -m idg_social_nav.plot_results --csv eval_results/run_eval_....csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from idg_social_nav.paths import CURVES_DIR, EVAL_DIR

COLLISION_WEIGHT = 1.0  # social cost = mean_intrusion_sum + weight * mean_collisions
_BLEND_RE = re.compile(r"^fixed_blend_p(\d+(?:\.\d+)?)$")

_POINT_STYLES = {
    "always_obey": dict(marker="s", color="C1"),
    "oracle": dict(marker="*", color="C2", markersize=13),
    "sac": dict(marker="D", color="C3"),
    "ppo": dict(marker="d", color="C5"),
}
_LLM_MARKERS = ("^", "v", "P", "X", "h", "8")


def newest_csv() -> Path:
    csvs = sorted(EVAL_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No eval CSVs found in {EVAL_DIR}. Run run_eval.py first.")
    return csvs[-1]


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def social_cost(row: dict) -> float:
    intrusion = _to_float(row.get("mean_intrusion_sum")) or 0.0
    collisions = _to_float(row.get("mean_collisions")) or 0.0
    return intrusion + COLLISION_WEIGHT * collisions


def blend_p(name: str) -> float | None:
    m = _BLEND_RE.match(name)
    return float(m.group(1)) if m else None


def plot_pareto(scenario: str, rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    blend_points = sorted(
        (blend_p(r["name"]), _to_float(r["mean_steps"]), social_cost(r))
        for r in rows if blend_p(r["name"]) is not None
    )
    if blend_points:
        xs = [x for _, x, _ in blend_points]
        ys = [y for _, _, y in blend_points]
        ax.plot(xs, ys, "-o", color="C0", markersize=4, lw=1.2,
                label="fixed blend (p sweep)", zorder=2)
        for p, x, y in blend_points:
            ax.annotate(f"{p:.1f}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="C0")

    llm_idx = 0
    for row in rows:
        name = row["name"]
        if blend_p(name) is not None:
            continue
        x = _to_float(row["mean_steps"])
        y = social_cost(row)
        if x is None:
            continue
        style = _POINT_STYLES.get(name)
        if style is None:
            style = dict(marker=_LLM_MARKERS[llm_idx % len(_LLM_MARKERS)],
                         color=f"C{4 + llm_idx % 5}")
            llm_idx += 1
        ax.plot([x], [y], linestyle="none", markersize=style.pop("markersize", 9),
                label=name, zorder=3, **style)

    ax.set_xlabel("mean steps to goal (timeouts at max)")
    ax.set_ylabel(f"social cost (intrusion sum + {COLLISION_WEIGHT:g} x collisions)")
    ax.set_title(f"Goal cost vs social cost - {scenario}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = CURVES_DIR / f"pareto_{scenario}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _breakdown_names(rows: list[dict]) -> list[str]:
    """Validator names for the bar chart:
    every non-sweep validator plus the
    mid-sweep fixed-blend point as the representative.
    """
    names = []
    blend_ps = []
    for row in rows:
        p = blend_p(row["name"])
        if p is None:
            if row["name"] not in names:
                names.append(row["name"])
        else:
            blend_ps.append(p)
    if blend_ps:
        rep = min(blend_ps, key=lambda p: abs(p - 0.5))
        names.append(f"fixed_blend_p{rep:.1f}")
    return names


def plot_breakdown(by_scenario: dict[str, list[dict]]) -> Path:
    scenarios = list(by_scenario)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.5 * len(scenarios), 4.5),
                             sharey=True, squeeze=False)

    bars = [
        ("good/dec %", "good_disobey_pct", 1.0, "C2"),
        ("bad/dec %", "bad_disobey_pct", 1.0, "C3"),
        ("precision %", "override_precision", 100.0, "C0"),
        ("recall %", "override_recall", 100.0, "C4"),
    ]
    width = 0.2

    for ax, scenario in zip(axes[0], scenarios, strict=True):
        rows = {r["name"]: r for r in by_scenario[scenario]}
        names = [n for n in _breakdown_names(by_scenario[scenario]) if n in rows]
        for j, (label, key, scale, color) in enumerate(bars):
            values = []
            for name in names:
                v = _to_float(rows[name].get(key))
                values.append(v * scale if v is not None else 0.0)
            xs = [i + (j - 1.5) * width for i in range(len(names))]
            ax.bar(xs, values, width=width, color=color,
                   label=label if scenario == scenarios[0] else None)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(scenario, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("percent")
    fig.legend(loc="upper right", fontsize=8)
    fig.suptitle("Disobedience quality and oracle agreement per validator")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = CURVES_DIR / "disobedience_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="eval CSV to plot (default: newest in eval_results/)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else newest_csv()
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}.")
    print(f"Plotting {len(rows)} rows from {csv_path}")

    CURVES_DIR.mkdir(exist_ok=True)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    for scenario, scenario_rows in by_scenario.items():
        print(f"wrote {plot_pareto(scenario, scenario_rows)}")
    print(f"wrote {plot_breakdown(by_scenario)}")


if __name__ == "__main__":
    main()
