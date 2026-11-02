"""Download and safely extract the version-checked SciFact data archive."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from scientific_evidence_rag.paths import PROCESSED_DIR, RAW_DIR, ROOT

MANIFEST_PATH = ROOT / "data" / "source_manifest.json"
ARCHIVE_NAME = "scifact_data.tar.gz"
ALLOWED_MEMBERS = {
    "data/corpus.jsonl": "corpus.jsonl",
    "data/claims_train.jsonl": "claims_train.jsonl",
    "data/claims_dev.jsonl": "claims_dev.jsonl",
    "data/claims_test.jsonl": "claims_test.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive() -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["archive_sha256"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / ARCHIVE_NAME
    if archive.exists():
        actual = sha256_file(archive)
        if actual != expected:
            raise ValueError(f"Existing archive checksum mismatch: {actual} != {expected}")
        return archive

    partial = archive.with_suffix(archive.suffix + ".part")
    request = urllib.request.Request(
        manifest["source"], headers={"User-Agent": "scientific-claim-evidence-rag/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    actual = sha256_file(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Downloaded archive checksum mismatch: {actual} != {expected}")
    partial.replace(archive)
    return archive


def extract_dataset(archive: Path) -> dict[str, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        for archive_name, local_name in ALLOWED_MEMBERS.items():
            member = members.get(archive_name)
            if member is None or not member.isfile():
                raise ValueError(
                    f"Expected regular file missing from SciFact archive: {archive_name}"
                )
            destination = PROCESSED_DIR / local_name
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {archive_name}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            with source, temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
            with destination.open(encoding="utf-8") as handle:
                counts[local_name] = sum(1 for line in handle if line.strip())

    expected_counts = {
        "corpus.jsonl": 5183,
        "claims_train.jsonl": 809,
        "claims_dev.jsonl": 300,
        "claims_test.jsonl": 300,
    }
    if counts != expected_counts:
        raise ValueError(f"SciFact record-count mismatch: got {counts}, expected {expected_counts}")
    return counts


def main() -> None:
    archive = download_archive()
    counts = extract_dataset(archive)
    print(f"Verified SciFact archive: {sha256_file(archive)}")
    print(f"Extracted counts: {counts}")
    print("Raw and processed dataset files are ignored by Git.")


if __name__ == "__main__":
    main()
