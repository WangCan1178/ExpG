#!/usr/bin/env python3
"""
aggregate_scores.py

Usage:
    python3 aggregate_scores.py /path/to/dir

Reads all JSON files under the given directory and groups them into four
categories according to the rules:
  1) multi_turn_base: files under a 'multi_turn' directory with 'base' in the filename
  2) multi_turn_other: other files under 'multi_turn'
  3) live_nonlive_rel_irrel: files under 'live' or 'non_live' whose filename
     contains 'irrelevance' or 'relevance'
  4) live_nonlive_other: other files under 'live' or 'non_live'

For each category it sums 'total_count' and 'correct_count' and computes accuracy.
If a JSON file lacks total_count/correct_count, it will be skipped with a warning.
"""
import os
import sys
import json
from collections import defaultdict


def categorize(file_path):
    """Return one of: multi_turn_base, multi_turn_other, live_nonlive_rel_irrel, live_nonlive_other, other"""
    path = os.path.normpath(file_path).lower()
    parts = [p for p in path.split(os.sep) if p]
    filename = os.path.basename(path)

    if 'multi_turn' in parts:
        if 'base' in filename:
            return 'multi_turn_base'
        else:
            return 'multi_turn_other'

    if 'live' in parts or 'non_live' in parts:
        if 'irrelevance' in filename or 'relevance' in filename:
            return 'live_nonlive_other'
        else:
            return 'live_nonlive_base'

    return 'other'


def process_directory(root_dir):
    sums = defaultdict(lambda: {'total_count': 0, 'correct_count': 0, 'files': []})
    warnings = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith('.json'):
                continue
            full = os.path.join(dirpath, fn)

            # Try to find a JSON object in the file that contains total_count & correct_count.
            # Files may be JSONL (multiple JSON objects per line), so parse line-by-line.
            t = None
            c = None
            try:
                with open(full, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            # skip malformed/non-json lines
                            continue
                        if isinstance(obj, dict) and 'total_count' in obj and 'correct_count' in obj:
                            try:
                                t = int(obj.get('total_count', 0))
                                c = int(obj.get('correct_count', 0))
                            except Exception:
                                t = int(obj.get('total_count', 0) or 0)
                                c = int(obj.get('correct_count', 0) or 0)
                            break
            except Exception as e:
                warnings.append(f"Failed to read {full}: {e}")
                continue

            if t is None or c is None:
                warnings.append(f"Skipping {full}: missing total_count or correct_count (no matching JSON object found)")
                continue

            cat = categorize(full)
            sums[cat]['total_count'] += t
            sums[cat]['correct_count'] += c
            sums[cat]['files'].append(full)

    return sums, warnings


def print_summary(sums, warnings):
    print('\nCategory summary:')
    # Print per-category
    grand_total = 0
    grand_correct = 0
    for cat, info in sums.items():
        total = info['total_count']
        correct = info['correct_count']
        acc = (correct / total) if total > 0 else None
        print(f"- {cat}: total_count={total}, correct_count={correct}, accuracy={acc}")
        grand_total += total
        grand_correct += correct

    # Print grand totals (aggregate across all categories)
    if grand_total > 0:
        grand_acc = grand_correct / grand_total
    else:
        grand_acc = None
    print(f"- ALL: total_count={grand_total}, correct_count={grand_correct}, accuracy={grand_acc}")

    if warnings:
        print('\nWarnings:')
        for w in warnings:
            print('- ' + w)


def main(argv):
    if len(argv) < 2:
        print('Usage: python3 aggregate_scores.py /path/to/dir')
        sys.exit(2)

    root = argv[1]
    if not os.path.isdir(root):
        print(f'Error: {root} is not a directory')
        sys.exit(2)

    sums, warnings = process_directory(root)
    print_summary(sums, warnings)


if __name__ == '__main__':
    main(sys.argv)
