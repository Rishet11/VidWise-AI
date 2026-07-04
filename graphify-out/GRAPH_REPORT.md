# Graph Report - .  (2026-07-04)

## Corpus Check
- Corpus is ~1,683 words - fits in a single context window. You may not need a graph.

## Summary
- 40 nodes · 60 edges · 13 communities (4 shown, 9 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `handle_all_events()` - 9 edges
2. `VidWise-AI` - 7 edges
3. `run_rag_chain()` - 6 edges
4. `show_history()` - 5 edges
5. `generate_summary()` - 4 edges
6. `main()` - 3 edges
7. `create_embeddings()` - 3 edges
8. `extract_youtube_id()` - 3 edges
9. `get_transcript()` - 3 edges
10. `message_alignment_style()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `LangChain` --semantically_similar_to--> `langchain`  [INFERRED] [semantically similar]
  README.md → requirements.txt
- `Streamlit` --semantically_similar_to--> `streamlit`  [INFERRED] [semantically similar]
  README.md → requirements.txt
- `FAISS` --semantically_similar_to--> `faiss-cpu`  [INFERRED] [semantically similar]
  README.md → requirements.txt
- `Gemini` --semantically_similar_to--> `google-generativeai`  [INFERRED] [semantically similar]
  README.md → requirements.txt
- `youtube-transcript-api` --semantically_similar_to--> `youtube-transcript-api`  [INFERRED] [semantically similar]
  README.md → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **VidWise-AI Core Technologies** — vidwise_ai_readme_langchain, vidwise_ai_readme_streamlit, vidwise_ai_readme_faiss, vidwise_ai_readme_gemini, vidwise_ai_readme_youtube_transcript_api [INFERRED 0.85]
- **Required Packages** — vidwise_ai_requirements_langchain, vidwise_ai_requirements_streamlit, vidwise_ai_requirements_faiss_cpu, vidwise_ai_requirements_google_generativeai, vidwise_ai_requirements_youtube_transcript_api [EXTRACTED 1.00]

## Communities (13 total, 9 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.40
Nodes (5): message_alignment_style(), Injects custom CSS for chat message alignment and styling,     with support for, Displays the chat history with right-left alignment for user and AI messages., show_context_chunks(), show_history()

### Community 1 - "Community 1"
Cohesion: 0.70
Nodes (4): build_prompt(), generate_response(), retrieve_documents(), run_rag_chain()

### Community 4 - "Community 4"
Cohesion: 0.67
Nodes (3): extract_youtube_id(), get_transcript(), handle_all_events()

### Community 5 - "Community 5"
Cohesion: 0.67
Nodes (3): Retrieval-Augmented Generation (RAG), Rishet Mehra, VidWise-AI

## Knowledge Gaps
- **8 isolated node(s):** `Retrieval-Augmented Generation (RAG)`, `Rishet Mehra`, `streamlit`, `langchain`, `faiss-cpu` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `handle_all_events()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 6`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `VidWise-AI` connect `Community 5` to `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 11`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `show_history()` connect `Community 0` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **What connects `Generates a summary of the transcript.`, `Injects custom CSS for chat message alignment and styling,     with support for`, `Displays the chat history with right-left alignment for user and AI messages.` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._