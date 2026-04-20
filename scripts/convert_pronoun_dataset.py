"""Convert the legacy MLX personalpronomen dataset to the JSONL format
consumed by regeln/personalpronomen/train_config.py.

Source format (JSON array, one object per pronoun instance):
    {
      "sentence_id": "...",
      "token_id": 6,
      "sentence_tokens": ["...", ...],
      "token": "es",
      "labels": {"person": 3, "gender": 3, "number": 2, "case": 3,
                 "polite": 0, "reflex": 0},
      "context_tokens": [...]
    }

Target format (JSONL, one object per line):
    {"text": "...", "pronoun_index": 6,
     "labels": {"person": "3", "gender": "Neut", "number": "Sing",
                "case": "Nom", "polite": "NONE", "reflex": "false"}}

Usage:
    python scripts/convert_pronoun_dataset.py
    python scripts/convert_pronoun_dataset.py --subset 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional

import ijson

DEFAULT_SOURCE_ROOT = Path(
    "/Users/ds/development/project/capazunda/personalpronomen/data/processed"
)
DEFAULT_TARGET_DIR = Path("regeln/personalpronomen/data")
DEFAULT_TARGET_LABELS = Path("regeln/personalpronomen/model/label_encoders.json")

HEADS = ["person", "gender", "number", "case", "polite", "reflex"]
SPLITS = [("train.json", "train.jsonl"), ("dev.json", "dev.jsonl"), ("test.json", "test.jsonl")]


def load_label_encoders(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_schema_compat(source: dict, target: dict) -> None:
    mismatches = []
    for head in HEADS:
        s_classes = source[head]["num_classes"]
        t_classes = target[head]["num_classes"]
        s_map = source[head]["label_to_id"]
        t_map = target[head]["label_to_id"]
        if s_classes != t_classes or s_map != t_map:
            mismatches.append(
                f"  {head}: source {s_classes}/{sorted(s_map.items())} "
                f"vs target {t_classes}/{sorted(t_map.items())}"
            )
    if mismatches:
        raise SystemExit(
            "Label schema mismatch — aborting to avoid corrupt labels:\n"
            + "\n".join(mismatches)
        )


def decode_label(head: str, int_id: int, source_encoders: dict) -> str:
    id_to_label = source_encoders[head]["id_to_label"]
    raw = id_to_label[str(int_id)]
    # The reflex head stores booleans in id_to_label; training code uses string keys.
    if isinstance(raw, bool):
        return "true" if raw else "false"
    return str(raw)


def stream_records(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def convert_split(
    source_path: Path,
    target_path: Path,
    source_encoders: dict,
    subset: Optional[int] = None,
    show_examples: int = 3,
) -> dict:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    per_head_counter = {h: Counter() for h in HEADS}
    total = 0
    skipped = 0
    examples_printed = 0

    with open(target_path, "w", encoding="utf-8") as out_f:
        for record in stream_records(source_path):
            if subset is not None and total >= subset:
                break

            sentence_tokens = record.get("sentence_tokens") or []
            token_id = record.get("token_id")
            labels_raw = record.get("labels") or {}

            if not sentence_tokens or token_id is None:
                skipped += 1
                continue
            if token_id < 0 or token_id >= len(sentence_tokens):
                skipped += 1
                continue

            text = " ".join(sentence_tokens)
            out_labels = {
                head: decode_label(head, labels_raw[head], source_encoders)
                for head in HEADS
                if head in labels_raw
            }
            if len(out_labels) != len(HEADS):
                skipped += 1
                continue

            for head, val in out_labels.items():
                per_head_counter[head][val] += 1

            obj = {"text": text, "pronoun_index": token_id, "labels": out_labels}
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if examples_printed < show_examples:
                print(f"  example {examples_printed + 1}: {json.dumps(obj, ensure_ascii=False)[:220]}")
                examples_printed += 1

            total += 1

    return {
        "total": total,
        "skipped": skipped,
        "per_head": {h: dict(c) for h, c in per_head_counter.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--target-labels", type=Path, default=DEFAULT_TARGET_LABELS)
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, only convert the first N records from each split (debug/speed).",
    )
    args = parser.parse_args()

    source_labels_path = args.source_root / "label_encoders.json"
    if not source_labels_path.exists():
        print(f"Missing source labels: {source_labels_path}", file=sys.stderr)
        return 1
    if not args.target_labels.exists():
        print(f"Missing target labels: {args.target_labels}", file=sys.stderr)
        return 1

    source_encoders = load_label_encoders(source_labels_path)
    target_encoders = load_label_encoders(args.target_labels)
    assert_schema_compat(source_encoders, target_encoders)
    print("✓ Label schemas match")

    grand_total = 0
    for source_name, target_name in SPLITS:
        source_path = args.source_root / source_name
        target_path = args.target_dir / target_name
        if not source_path.exists():
            print(f"⚠ Missing source: {source_path} — skipping")
            continue
        print(f"\n→ {source_path}  ->  {target_path}")
        stats = convert_split(source_path, target_path, source_encoders, subset=args.subset)
        grand_total += stats["total"]
        print(f"  wrote {stats['total']} instances  (skipped {stats['skipped']})")
        for head in HEADS:
            counts = stats["per_head"][head]
            total = sum(counts.values()) or 1
            parts = ", ".join(
                f"{k}={counts[k]}({counts[k] / total:.1%})"
                for k in sorted(counts, key=counts.get, reverse=True)
            )
            print(f"    {head:<7}: {parts}")

    print(f"\n✓ Converted {grand_total} total instances across {len(SPLITS)} splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
