#!/usr/bin/env python3
"""Retrofit the shared reporting preset into existing frozen models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from output_preset import apply_reporting_preset, infer_building_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the shared reporting preset to frozen models.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("frozen_models"),
        help="Directory containing frozen epJSON models.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_frozen_detailed.epJSON",
        help="Filename glob to patch.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    models_dir = args.models_dir if args.models_dir.is_absolute() else script_dir / args.models_dir

    model_paths = sorted(models_dir.glob(args.glob))
    if not model_paths:
        raise SystemExit(f"No models matched {args.glob} in {models_dir}")

    for model_path in model_paths:
        with open(model_path, "r", encoding="utf-8") as handle:
            model = json.load(handle)

        building_key = infer_building_key(model_path)
        counts = apply_reporting_preset(model, building_key)

        with open(model_path, "w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2)
            handle.write("\n")

        print(
            f"{model_path.name}: {counts['variables']} vars, "
            f"{counts['meters']} meters, {counts['summary_reports']} summaries"
        )


if __name__ == "__main__":
    main()
