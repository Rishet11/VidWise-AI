# Launch package

Live: https://huggingface.co/spaces/Rishet11/vidwise

Status: evaluation harness is built (`benchmarks/`), human-labeled accuracy numbers are not published yet. Do not claim accuracy percentages until that lands.

## X post (under 280 chars)

> Built VidWise: paste up to 6 YouTube videos, ask a question, get an answer where every claim links to the exact second it came from. Click a citation, it jumps you straight to that moment in the video. Live: https://huggingface.co/spaces/Rishet11/vidwise

(258 chars)

## LinkedIn variant

I built VidWise, a tool for researching across multiple YouTube videos at once.

You add up to 6 videos, ask one question, and it answers with citations, each claim links to the exact second in the source video, with the transcript snippet right there to check.

The hard part wasn't the LLM call. It was making citations trustworthy: timestamped chunking, retrieval across merged video indexes, and keeping every claim traceable back to a real second instead of a plausible-sounding paraphrase.

Still a student project, still early. Evaluation harness is built, I haven't published accuracy numbers yet because I only report what's measured.

Live: https://huggingface.co/spaces/Rishet11/vidwise

## 15-second demo shot list

- 0-3s: Paste 3 YouTube video links into the input fields.
- 3-6s: Type a cross-video question into the question box.
- 6-12s: Answer appears with claim text and citation chips (timestamp links); one chip's evidence panel expands to show the transcript snippet.
- 12-15s: Click a citation chip, cut to YouTube opening at that exact second (the `&t=Ns` link).

Record only after a cold-browser run of the live Space works end to end (first load can take 2-3 minutes to wake).
