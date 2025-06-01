import os
import json
from pathlib import Path

def fix_dataset_json(json_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    changed = False

    # Fix labels (convert 'background': 0 to '0': 'background')
    if data.get("labels") and not all(k.isdigit() for k in data["labels"].keys()):
        data["labels"] = {str(v): k for k, v in data["labels"].items()}
        changed = True

    # Ensure modality and channel_names are both present
    if "modality" in data and "channel_names" not in data:
        data["channel_names"] = data["modality"]
        changed = True
    elif "channel_names" in data and "modality" not in data:
        data["modality"] = data["channel_names"]
        changed = True

    if changed:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Fixed: {json_path}")

def walk_and_fix(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if 'dataset.json' in files:
            fix_dataset_json(Path(root) / 'dataset.json')

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_all_datasets.py /path/to/root_folder")
    else:
        walk_and_fix(sys.argv[1])
