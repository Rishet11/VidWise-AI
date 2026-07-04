"""Private Streamlit editor for Rishet's personal evaluation labels."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

DATASET = Path(__file__).with_name("dataset.jsonl")


def load_rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rows(rows):
    temporary = DATASET.with_suffix(".tmp")
    temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    temporary.replace(DATASET)


st.set_page_config(page_title="VidWise labeler", layout="wide")
st.title("VidWise evaluation-set labeler")
st.warning("Private tool: labels must come from watching the videos. Do not paste full transcripts or use generated labels as truth.")
rows = load_rows()
row_id = st.selectbox("Question", [row["id"] for row in rows])
index = next(i for i, row in enumerate(rows) if row["id"] == row_id)
row = rows[index]

question = st.text_input("Question", row["question"])
video_ids = st.text_area("Applicable video IDs (one per line)", "\n".join(row.get("video_ids", [])))
st.markdown("Relevant segments: add only short snippets needed to verify the time range.")
segments_text = st.text_area(
    "Segments JSON",
    json.dumps(row.get("relevant_segments", []), indent=2),
    height=260,
    help='[{"video_id":"…","start":12.3,"end":25.0,"snippet":"short quote"}]',
)
claims_text = st.text_area("Atomic expected claims JSON", json.dumps(row.get("expected_claims", []), indent=2), height=160)
negative = st.checkbox("Insufficient-evidence / negative question", row.get("negative", False))
split = st.selectbox("Split", ["dev", "held_out"], index=0 if row.get("split") == "dev" else 1)
confirmed = st.checkbox("I personally watched every cited range and verified this row")

if st.button("Validate and save row", type="primary"):
    try:
        segments = json.loads(segments_text)
        claims = json.loads(claims_text)
        ids = [item.strip() for item in video_ids.splitlines() if item.strip()]
        if not ids:
            raise ValueError("At least one applicable video ID is required")
        if not negative and (not segments or not claims):
            raise ValueError("Positive questions require relevant segments and expected claims")
        for segment in segments:
            if not {"video_id", "start", "end", "snippet"} <= set(segment):
                raise ValueError("Every segment needs video_id, start, end, and snippet")
            if segment["video_id"] not in ids or float(segment["start"]) >= float(segment["end"]):
                raise ValueError("Segment video/range is invalid")
            if len(segment["snippet"].split()) > 25:
                raise ValueError("Keep each quoted snippet at 25 words or fewer")
        if not confirmed:
            raise ValueError("Personal verification checkbox is required")
        rows[index] = {
            **row, "question": question.strip(), "video_ids": ids,
            "relevant_segments": segments, "expected_claims": claims,
            "negative": negative, "split": split, "label_status": "human_verified",
        }
        save_rows(rows)
        st.success(f"Saved {row_id} as human_verified")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        st.error(str(exc))

verified = sum(row.get("label_status") == "human_verified" for row in load_rows())
st.progress(verified / len(rows), text=f"{verified}/{len(rows)} questions personally verified")

