import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from web_agent_site.utils import FULL_FILE_PATH


def iter_json_array(path: str, chunk_size: int = 1024 * 1024) -> Iterable[Dict[str, Any]]:
    started_array = False
    collecting = False
    in_string = False
    escaped = False
    depth = 0
    object_chunks = []

    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for char in chunk:
                if not started_array:
                    if char.isspace():
                        continue
                    if char != "[":
                        raise ValueError(f"{path} must contain a JSON array")
                    started_array = True
                    continue

                if not collecting:
                    if char.isspace() or char == ",":
                        continue
                    if char == "]":
                        return
                    if char != "{":
                        raise ValueError(f"Expected product object in {path}")
                    collecting = True
                    depth = 1
                    in_string = False
                    escaped = False
                    object_chunks = [char]
                    continue

                object_chunks.append(char)
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        item = json.loads("".join(object_chunks))
                        if isinstance(item, dict):
                            yield item
                        collecting = False
                        object_chunks = []


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(v) for v in value.values())
    return str(value)


def option_text(product: Dict[str, Any]) -> str:
    options = product.get("customization_options") or product.get("options") or {}
    if not isinstance(options, dict):
        return ""

    parts = []
    for option_name, option_contents in options.items():
        if not option_contents:
            continue
        if isinstance(option_contents, list):
            values = []
            for option_content in option_contents:
                if isinstance(option_content, dict):
                    values.append(normalize_text(option_content.get("value")))
                else:
                    values.append(normalize_text(option_content))
            option_contents_text = ", ".join(v for v in values if v)
        else:
            option_contents_text = normalize_text(option_contents)
        if option_contents_text:
            parts.append(f"{option_name}: {option_contents_text}")
    return ", and ".join(parts)


def make_doc(product: Dict[str, Any]) -> Dict[str, str]:
    asin = str(product.get("asin") or "").strip()
    contents = " ".join(
        part
        for part in [
            normalize_text(product.get("name") or product.get("Title")),
            normalize_text(product.get("full_description") or product.get("Description")),
            normalize_text(product.get("small_description") or product.get("BulletPoints")),
            option_text(product),
        ]
        if part
    ).lower()
    return {"id": asin, "contents": contents}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        choices=["resources_100", "resources", "resources_1k", "resources_100k", "all"],
        default="all",
        help="Which resources directory to write. Use resources_100k for a 100k-only build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_output_specs = [
        ("resources_100", 100),
        ("resources", None),
        ("resources_1k", 1000),
        ("resources_100k", 100000),
    ]
    output_specs = (
        all_output_specs
        if args.output == "all"
        else [spec for spec in all_output_specs if spec[0] == args.output]
    )
    max_limit = None if any(limit is None for _, limit in output_specs) else max(limit for _, limit in output_specs)
    handles = []
    try:
        for dirname, _ in output_specs:
            output_dir = SCRIPT_DIR / dirname
            output_dir.mkdir(parents=True, exist_ok=True)
            handles.append(open(output_dir / "documents.jsonl", "w", encoding="utf-8"))

        for index, product in enumerate(tqdm(iter_json_array(FULL_FILE_PATH)), start=1):
            doc = make_doc(product)
            if not doc["id"]:
                continue
            line = json.dumps(doc) + "\n"
            for handle, (_, limit) in zip(handles, output_specs):
                if limit is None or index <= limit:
                    handle.write(line)
            if max_limit is not None and index >= max_limit:
                break
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
