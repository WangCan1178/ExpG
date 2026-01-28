#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare API-Bank results: baseline (no suffix) vs mem / fewshot / draft / m0,
focusing on correct and incorrect coverage. Each method has 0/1/2 samples;
if any sample of the same task succeeds, we treat it as success.
"""
from __future__ import annotations

import json
from pathlib import Path
import math
from typing import Dict, Iterable, List
import matplotlib.pyplot as plt

BASE_DIR = Path("/Users/xiaocan/Toolmem/experiment/api-bank")

FILE_GROUPS = {
    "base": [
        "qwen3_8b_2.jsonl",
    ],
    "mem": [
        "qwen3_8b_mem_2.jsonl",
    ],
    "fewshot": [
        "qwen3_8b_fewshot_2.jsonl",
    ],
    "draft": [
        "qwen3-8b_draft_2.jsonl",
    ],
    "m0": [
        "qwen3-8b_m0_2.jsonl",
    ],
}


def read_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open() as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as exc:  # pragma: no cover - debug info
                print(f"Parse failed {path} line {line_idx + 1}: {exc}")
    return items


def make_key(item: dict, fallback_index: int) -> str:
    file_name = str(item.get("file") or "unknown")
    sample_id = item.get("sample_id")
    if sample_id is None:
        sample_id = f"line{fallback_index}"
    return f"{file_name}|{sample_id}"


def aggregate_method(files: Iterable[str]) -> Dict[str, bool]:
    """Aggregate multiple samples of one method. Key = file|sample_id; value = correctness."""
    collected: Dict[str, List[bool]] = {}
    def is_level_1_or_2(name: str) -> bool:
        lower = name.lower()
        return "level-1" in lower or "level-2" in lower

    for fname in files:
        path = BASE_DIR / fname
        if not path.exists():
            print(f"Warning: missing file {path}")
            continue
        for idx, item in enumerate(read_jsonl(path)):
            file_field = str(item.get("file", ""))
            if not is_level_1_or_2(file_field):
                continue
            key = make_key(item, idx)
            ok = bool(item.get("correct", False))
            collected.setdefault(key, []).append(ok)
    return {k: any(v_list) for k, v_list in collected.items()}


def compare_to_base(base: Dict[str, bool], method: Dict[str, bool]) -> Dict[str, int]:
    """Coverage stats relative to the no-method baseline."""
    keys = set(base) | set(method)
    base_right = {k for k in keys if base.get(k, False)}
    method_right = {k for k in keys if method.get(k, False)}
    base_wrong = keys - base_right
    method_wrong = keys - method_right

    return {
        "total": len(keys),
        "right_common": len(base_right & method_right),
        "right_method_only": len(method_right - base_right),  # method succeeds, base fails
        "right_base_only": len(base_right - method_right),    # base succeeds, method fails
        "wrong_common": len(base_wrong & method_wrong),
        "wrong_method_only": len(method_wrong - base_wrong),  # base succeeds, method fails
        "wrong_base_only": len(base_wrong - method_wrong),    # base fails, method succeeds
    }


def compute_deltas(base: Dict[str, bool], method: Dict[str, bool]) -> Dict[str, set]:
    """
    Compute method gains/losses relative to baseline.
    right_gain: method correct while baseline incorrect.
    error: baseline correct while method incorrect.
    """
    keys = set(base) | set(method)
    base_right = {k for k in keys if base.get(k, False)}
    method_right = {k for k in keys if method.get(k, False)}
    right_gain = method_right - base_right
    error = base_right - method_right
    return {"right_gain": right_gain, "error": error}


def build_mem_vs_other(mem_delta: Dict[str, set], other_delta: Dict[str, set]) -> Dict[str, Dict[str, int]]:
    """
    Overlap between mem and another method on gains/errors relative to baseline.
    right: overlap of right_gain sets; wrong: overlap of error sets.
    """
    mem_right = mem_delta["right_gain"]
    mem_error = mem_delta["error"]
    other_right = other_delta["right_gain"]
    other_error = other_delta["error"]

    return {
        "right": {
            "common": len(mem_right & other_right),
            "mem_only": len(mem_right - other_right),
            "other_only": len(other_right - mem_right),
        },
        "wrong": {
            "common": len(mem_error & other_error),
            "mem_only": len(mem_error - other_error),
            "other_only": len(other_error - mem_error),
        },
    }


def _circle_distance(r1: float, r2: float, target_common: int, mem_total: int, other_total: int) -> float:
    if mem_total == 0 or other_total == 0 or target_common == 0:
        return r1 + r2 + 0.5
    mem_area = math.pi * r1 * r1
    target_area = target_common * (mem_area / mem_total)

    def circle_intersection_area(d: float) -> float:
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return math.pi * min(r1, r2) ** 2
        d1 = (r1**2 - r2**2 + d**2) / (2 * d)
        d2 = d - d1
        area1 = r1**2 * math.acos(d1 / r1) - d1 * math.sqrt(max(r1**2 - d1**2, 0.0))
        area2 = r2**2 * math.acos(d2 / r2) - d2 * math.sqrt(max(r2**2 - d2**2, 0.0))
        return area1 + area2

    lo = max(0.0, abs(r1 - r2))
    hi = r1 + r2
    for _ in range(40):
        mid = (lo + hi) / 2
        area = circle_intersection_area(mid)
        if area > target_area:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def plot_overlaps(overlaps: Dict[str, Dict[str, Dict[str, int]]], save_path: Path) -> None:
    methods = list(overlaps.keys())
    fig, axes = plt.subplots(2, len(methods), figsize=(5 * len(methods), 8))
    fig.suptitle("mem overlap comparison", fontsize=16, fontweight="bold", y=0.98)

    mem_color = "#2196F3"
    other_colors = {
        "m0": "#FF9800",
        "fewshot": "#4CAF50",
        "draft": "#9C27B0",
    }
    alpha_value = 0.5

    max_size = 0
    for method, stats in overlaps.items():
        for part in ("right", "wrong"):
            data = stats[part]
            total_mem = data["mem_only"] + data["common"]
            total_other = data["other_only"] + data["common"]
            max_size = max(max_size, total_mem, total_other)
    max_radius = 2.3
    scale = max_radius / (max_size ** 0.5) if max_size > 0 else 1.0

    for col, method in enumerate(methods):
        for row, part in enumerate(("right", "wrong")):
            ax = axes[row, col]
            data = overlaps[method][part]
            common = data["common"]
            mem_only = data["mem_only"]
            other_only = data["other_only"]
            mem_total = mem_only + common
            other_total = other_only + common
            mem_r = (mem_total ** 0.5) * scale if mem_total > 0 else 0.1
            other_r = (other_total ** 0.5) * scale if other_total > 0 else 0.1

            distance = _circle_distance(mem_r, other_r, common, mem_total or 1, other_total or 1)
            mem_cx = -distance / 2
            other_cx = distance / 2
            cy = 0

            mem_circle = plt.Circle((mem_cx, cy), mem_r, color=mem_color, alpha=alpha_value, label="mem" if row == 0 and col == 0 else "")
            other_circle = plt.Circle((other_cx, cy), other_r, color=other_colors.get(method, "#999999"), alpha=alpha_value, label=method if row == 0 else "")
            ax.add_patch(mem_circle)
            ax.add_patch(other_circle)

            if mem_total > 0:
                ax.text(mem_cx, cy, f"{mem_total}", ha="center", va="center", fontsize=11, fontweight="bold", color="white", bbox=dict(boxstyle="round", facecolor="black", alpha=0.3))
            if other_total > 0:
                ax.text(other_cx, cy, f"{other_total}", ha="center", va="center", fontsize=11, fontweight="bold", color="white", bbox=dict(boxstyle="round", facecolor="black", alpha=0.3))
            if common > 0:
                ax.text((mem_cx + other_cx) / 2, cy, f"common:{common}", ha="center", va="center", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"))

            ax.text(mem_cx, cy - mem_r - 0.25, "mem", ha="center", va="top", fontsize=10, fontweight="bold")
            ax.text(other_cx, cy - other_r - 0.25, method, ha="center", va="top", fontsize=10, fontweight="bold")

            max_r = max(mem_r, other_r)
            margin = max_r * 0.5
            ax.set_xlim(-distance / 2 - max_r - margin, distance / 2 + max_r + margin)
            ax.set_ylim(-max_r - margin, max_r + margin * 0.8)
            ax.set_aspect("equal")
            title = f"mem vs {method} right" if part == "right" else f"mem vs {method} wrong"
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=mem_color, lw=8, alpha=alpha_value, label="mem"),
    ]
    for m, c in other_colors.items():
        handles.append(plt.Line2D([0], [0], color=c, lw=8, alpha=alpha_value, label=m))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.97), frameon=True, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {save_path}")


def print_summary(name: str, stats: Dict[str, int]) -> None:
    total = stats["total"] or 1
    def pct(n: int) -> str:
        return f"{n} ({n / total:.2%})"

    print(f"\n=== {name} vs baseline ===")
    print(f"total tasks: {stats['total']}")
    print(f"right-common: {pct(stats['right_common'])}")
    print(f"right-method-only (gain): {pct(stats['right_method_only'])}")
    print(f"right-base-only: {pct(stats['right_base_only'])}")
    print(f"wrong-common: {pct(stats['wrong_common'])}")
    print(f"wrong-method-only (loss): {pct(stats['wrong_method_only'])}")
    print(f"wrong-base-only: {pct(stats['wrong_base_only'])}")


def print_overlap_report(mem_delta: Dict[str, set], other_name: str, overlap: Dict[str, Dict[str, int]], other_delta: Dict[str, set]) -> None:
    right_gain = len(other_delta["right_gain"])
    error = len(other_delta["error"])
    right_common = overlap["right"]["common"]
    error_common = overlap["wrong"]["common"]
    print(f"\n--- {other_name} gains/losses ---")
    print(f"right_gain vs baseline: {right_gain}")
    print(f"  overlap with mem right_gain: {right_common}")
    print(f"errors vs baseline: {error}")
    print(f"  overlap with mem errors: {error_common}")


def main() -> None:
    print("Loading results...")
    base_res = aggregate_method(FILE_GROUPS["base"])
    method_results = {name: aggregate_method(files) for name, files in FILE_GROUPS.items() if name != "base"}

    print(f"Baseline task count: {len(base_res)}")
    overlaps: Dict[str, Dict[str, Dict[str, int]]] = {}
    deltas: Dict[str, Dict[str, set]] = {}
    for name, res in method_results.items():
        stats = compare_to_base(base_res, res)
        print_summary(name, stats)
        deltas[name] = compute_deltas(base_res, res)

    if "mem" not in deltas:
        print("mem method not found, skip plotting.")
        return

    for name, delta in deltas.items():
        if name == "mem":
            continue
        overlaps[name] = build_mem_vs_other(deltas["mem"], delta)
        print_overlap_report(deltas["mem"], name, overlaps[name], delta)

    plot_overlaps(overlaps, Path("/Users/xiaocan/Toolmem/mem_comparison.png"))


if __name__ == "__main__":
    main()
