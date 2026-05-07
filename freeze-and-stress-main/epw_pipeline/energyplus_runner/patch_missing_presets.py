#!/usr/bin/env python3
"""One-off script: apply reporting preset to frozen models that are missing Output:SQLite."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_preset import apply_reporting_preset, infer_building_key

FROZEN_DIR = Path(__file__).resolve().parent / "frozen_models"

patched = []
for epjson_path in sorted(FROZEN_DIR.glob("*_frozen_detailed.epJSON")):
    with open(epjson_path, encoding="utf-8") as f:
        model = json.load(f)

    if "Output:SQLite" in model:
        continue

    building_key = infer_building_key(epjson_path)
    counts = apply_reporting_preset(model, building_key)

    with open(epjson_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    patched.append(f"  {epjson_path.name}  ({building_key}) -> {counts}")

if patched:
    print(f"Patched {len(patched)} files:")
    print("\n".join(patched))
else:
    print("All frozen models already have Output:SQLite. Nothing to do.")
