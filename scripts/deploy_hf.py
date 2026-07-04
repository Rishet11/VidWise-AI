#!/usr/bin/env python3
"""Explicit, credentialed push to an existing/new Docker HF Space."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="owner/space-name")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="Required because this changes external state")
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN is missing")
    if "/" not in args.space:
        parser.error("--space must be owner/name")
    if not args.confirm:
        parser.error("Review the target, then repeat with --confirm")
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=[".git/**", ".env", ".venv/**", "logs/**", "data/transcripts/**", "graphify-out/**", "__pycache__/**"],
        commit_message="Deploy VidWise Docker Space",
    )
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()

