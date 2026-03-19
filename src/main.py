from __future__ import annotations

import argparse
from pathlib import Path

from generator import IntentionShiftGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a multi-turn intention shift benchmark dataset.")
    parser.add_argument("--output", type=str, default="../data/benchmark_dataset.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_shift_turns", type=int, default=3)
    args = parser.parse_args()

    generator = IntentionShiftGenerator(seed=args.seed)
    output = Path(__file__).resolve().parent / args.output
    generator.export_dataset(output, num_shift_turns=args.num_shift_turns)
    print(f"Dataset written to {output}")


if __name__ == "__main__":
    main()
