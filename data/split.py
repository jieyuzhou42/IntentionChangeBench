import json
from pathlib import Path

# split.py 所在目录，也就是 IntentionChangeBench/data
script_dir = Path(__file__).resolve().parent

input_path = script_dir / "webshop_simulated_dataset.json"
output_dir = script_dir / "webshop_tasks_split"
output_dir.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Expected the dataset to be a list of task instances.")

for i, task in enumerate(data, start=1):
    instance_id = task.get("instance_id", f"task_{i:03d}")

    safe_instance_id = "".join(
        c if c.isalnum() or c in ("_", "-", ".") else "_"
        for c in instance_id
    )

    output_path = output_dir / f"{i:02d}_{safe_instance_id}_new.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(task, f, ensure_ascii=False, indent=2)

print(f"Done. Split {len(data)} tasks into folder: {output_dir}")