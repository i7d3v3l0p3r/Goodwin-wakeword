"""Build model release assets and the catalog consumed by clients."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from shutil import copyfile


MODELS = Path(__file__).resolve().parents[1] / "models"
SUPPORT_FILES = ("LICENSE", "NOTICE")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package(tag: str, output: Path) -> None:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Expected a release tag in the form vMAJOR.MINOR.PATCH")

    catalog = json.loads((MODELS / "catalog.json").read_text(encoding="utf-8"))
    filenames = {entry["filename"] for entry in catalog}
    ids = {entry["id"] for entry in catalog}
    if not catalog or len(filenames) != len(catalog) or len(ids) != len(catalog):
        raise ValueError("Model catalog must have unique IDs and filenames")

    model_files = {path.name for path in MODELS.iterdir() if path.suffix in {".onnx", ".tflite"}}
    if filenames != model_files:
        raise ValueError(f"Model files and catalog differ: {sorted(filenames ^ model_files)}")
    for filename in filenames | set(SUPPORT_FILES):
        path = MODELS / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty release asset: {filename}")

    # A fresh directory prevents an older package's files from entering a new release.
    output.mkdir(parents=True, exist_ok=False)
    models = []
    for entry in catalog:
        filename = entry["filename"]
        asset = output / filename
        copyfile(MODELS / filename, asset)
        models.append({**entry, "size_bytes": asset.stat().st_size, "sha256": sha256(asset)})
    for filename in SUPPORT_FILES:
        copyfile(MODELS / filename, output / filename)

    metadata = {"schema_version": 1, "tag": tag, "models": models}
    with (output / "metadata.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    assets = sorted(output.iterdir())
    with (output / "SHA256SUMS").open("w", encoding="ascii", newline="\n") as stream:
        for path in assets:
            stream.write(f"{sha256(path)}  {path.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory for release files")
    args = parser.parse_args()
    package(args.tag, args.out)
