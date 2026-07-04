# VidWise evaluation methodology

Status: implementation draft; publish only after human labels and measured runs exist.

VidWise separates retrieval quality from answer grounding. A frozen public-video corpus is labelled at the segment level, retrieval is scored by overlap with those time ranges, and every generated atomic claim is manually checked against its cited snippet. Negative questions test abstention. Wilson intervals accompany rates; three repeated runs expose latency variance. A naive dense-retrieval configuration is compared with multi-query expansion, listwise reranking, and both together.

The critical validity check is human review. An LLM judge may accelerate inspection, but its agreement with the human labels is reported and it never defines ground truth. A second labeler independently reviews at least ten questions in v2; low raw overlap triggers adjudication and a protocol revision. Failure analysis names mechanisms—such as caption boundary drift or reranker displacement—rather than grouping every error under “hallucination.”

Full results remain unpublished until the dataset is personally labelled and the run artifact can reproduce every table cell.

