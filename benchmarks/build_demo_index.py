#!/usr/bin/env python3
"""Build a read-only FAISS artifact from cached, approved demo transcripts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.embeddings import build_index
from core.transcript import load_cached_transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data/demo/manifest.json"))
    parser.add_argument("--output", default=str(ROOT / "data/demo/index"))
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if not 5 <= len(manifest["videos"]) <= 10:
        parser.error("Demo corpus must contain 5–10 approved public videos")
    transcripts = [load_cached_transcript(item["video_id"]) for item in manifest["videos"]]
    missing = [item["video_id"] for item, transcript in zip(manifest["videos"], transcripts) if transcript is None]
    if missing:
        parser.error("Fetch/cache these demo transcripts first: " + ", ".join(missing))
    index = build_index(transcripts)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import faiss

    faiss.write_index(index.index, str(output / "corpus.faiss"))
    (output / "chunks.json").write_text(json.dumps([chunk.__dict__ for chunk in index.chunks]), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

