#!/usr/bin/env python3
import os
import json
import sys
from pathlib import Path

# Adjust Python path to load core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.transcript import get_transcript, load_cached_transcript
from core.llm import QuestionLLM

VIDEOS = [
    {"video_id": "DOtCl5PU8F0", "title": "How to Evaluate Startup Ideas", "url": "https://www.youtube.com/watch?v=DOtCl5PU8F0"},
    {"video_id": "z1iF1c8w5Lg", "title": "How To Talk To Users", "url": "https://www.youtube.com/watch?v=z1iF1c8w5Lg"},
    {"video_id": "12D8zEdOPYo", "title": "How to Build Products Users Love", "url": "https://www.youtube.com/watch?v=12D8zEdOPYo"},
    {"video_id": "8pNxKX1SUGE", "title": "All About Pivoting", "url": "https://www.youtube.com/watch?v=8pNxKX1SUGE"},
    {"video_id": "1hHMwLxN6EM", "title": "How to Plan an MVP", "url": "https://www.youtube.com/watch?v=1hHMwLxN6EM"}
]

VIDEO_IDS = [v["video_id"] for v in VIDEOS]

def fetch_and_cache_transcripts():
    print("Fetching and caching transcripts...")
    transcripts = {}
    for video in VIDEOS:
        vid = video["video_id"]
        print(f"Fetching transcript for {vid} ({video['title']})...")
        t = get_transcript(vid, allow_local=True)
        transcripts[vid] = t
        print(f"  Fetched. Title: {t.title}, segments: {len(t.segments)}, source: {t.source}")
    return transcripts

def write_manifest():
    print("Writing data/demo/manifest.json...")
    manifest_path = ROOT / "data/demo/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "approved",
        "videos": VIDEOS,
        "policy": "Add 5–10 public founder-interview videos only after checking transcript rights and stability. Do not commit raw transcripts."
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Manifest written.")

def build_index():
    print("Running benchmarks/build_demo_index.py...")
    import subprocess
    res = subprocess.run([sys.executable, str(ROOT / "benchmarks/build_demo_index.py")], capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    if res.returncode != 0:
        print("STDERR:")
        print(res.stderr)
        raise RuntimeError("Failed to build index")

def prefill_dataset(transcripts):
    print("Prefilling dataset...")
    dataset_path = ROOT / "benchmarks/dataset.jsonl"
    
    # Read the existing dataset rows
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    
    # Format transcripts to pass to Gemini
    formatted_parts = []
    for vid, t in transcripts.items():
        formatted_parts.append(f"=== VIDEO ID: {vid} | TITLE: {t.title} ===")
        for seg in t.segments:
            formatted_parts.append(f"[{seg.start:.1f} - {seg.end:.1f}] {seg.text}")
    formatted_transcripts = "\n".join(formatted_parts)
    
    # Define JSON schema for Gemini response
    schema = {
        "type": "OBJECT",
        "properties": {
            "relevant_segments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "video_id": {"type": "STRING"},
                        "start": {"type": "NUMBER"},
                        "end": {"type": "NUMBER"},
                        "snippet": {"type": "STRING"}
                    },
                    "required": ["video_id", "start", "end", "snippet"]
                }
            },
            "expected_claims": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["relevant_segments", "expected_claims"]
    }
    
    # Retry mechanism for Gemini calls and model selection
    import time
    
    updated_rows = []
    for row in rows:
        qid = row["id"]
        qtext = row["question"]
        is_negative = row.get("negative", False)
        
        print(f"Processing {qid}: {qtext} (Negative: {is_negative})")
        
        # All questions must list all 5 video IDs
        row["video_ids"] = VIDEO_IDS
        row["label_status"] = "needs_rishet_review"
        
        if is_negative:
            row["relevant_segments"] = []
            row["expected_claims"] = []
        else:
            prompt = f"""You are a startup school evaluation assistant.
Here are the transcripts for 5 YC Startup School videos:

{formatted_transcripts}

We are constructing an evaluation dataset. Based on the transcripts, identify the relevant segment(s) and expected atomic claims for this question:
Question: "{qtext}"

Guidelines:
1. `relevant_segments`: Identify the exact parts of the transcripts where this question is directly answered.
   For each segment:
   - `video_id` MUST be one of: {", ".join(VIDEO_IDS)}
   - `start` (float/int) is the start timestamp in seconds.
   - `end` (float/int) is the end timestamp in seconds.
   - `snippet` is the verbatim text from the transcript representing this segment. It MUST be 25 words or fewer.
2. `expected_claims`: A list of short, atomic declarative sentences (claims) that a correct answer to this question MUST contain.

Return the result conforming exactly to the JSON schema.
"""
            # Retry mechanism for Gemini calls
            res = None
            models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
            for model_name in models_to_try:
                print(f"  Attempting with model: {model_name}")
                llm = QuestionLLM(model=model_name)
                for attempt in range(5):
                    try:
                        res = llm.generate_json(prompt, schema)
                        break
                    except Exception as e:
                        print(f"    Attempt {attempt + 1} failed: {e}")
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "limit" in str(e).lower():
                            print("    Rate limit hit. Sleeping 35 seconds before retry...")
                            time.sleep(35)
                        else:
                            time.sleep(5)
                if res is not None:
                    break
            
            if res is None:
                raise RuntimeError(f"Failed to generate output for {qid} after trying all models and retries.")
            
            # Post-process and validate segments
            valid_segments = []
            for seg in res.get("relevant_segments", []):
                vid = seg.get("video_id")
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                snippet = str(seg.get("snippet", ""))
                
                # Check video ID
                if vid not in VIDEO_IDS:
                    print(f"  Warning: video ID {vid} not in {VIDEO_IDS}. Skipping segment.")
                    continue
                
                # Check start/end
                if start >= end:
                    print(f"  Warning: start ({start}) >= end ({end}). Adjusting end.")
                    end = start + 5.0
                
                # Truncate snippet word count to <= 25 words
                words = snippet.split()
                if len(words) > 25:
                    print(f"  Warning: Snippet too long ({len(words)} words). Truncating.")
                    snippet = " ".join(words[:25])
                
                valid_segments.append({
                    "video_id": vid,
                    "start": start,
                    "end": end,
                    "snippet": snippet
                })
            
            row["relevant_segments"] = valid_segments
            row["expected_claims"] = res.get("expected_claims", [])
            print(f"  Generated {len(valid_segments)} relevant segments and {len(row['expected_claims'])} expected claims.")
            # Sleep between queries to avoid rate limits
            time.sleep(10)
            
        updated_rows.append(row)
        
    # Write back to dataset.jsonl
    dataset_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in updated_rows) + "\n", encoding="utf-8")
    print("dataset.jsonl written successfully.")

def main():
    transcripts = fetch_and_cache_transcripts()
    write_manifest()
    build_index()
    prefill_dataset(transcripts)

if __name__ == "__main__":
    main()
