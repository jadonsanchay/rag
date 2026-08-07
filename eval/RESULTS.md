# Phase A results — retrieval baseline & chunking experiments

**Corpus:** `fastapi` pinned at tag `0.115.0` (SHA `40e33e49`), restricted to
`fastapi/` + `docs/en/` — 201 files (42 Python, 144 Markdown).

**Golden set:** 30 hand-labelled questions in `golden.jsonl`, tagged `pinpoint`
(12), `behavioral` (12), `architectural` (6). A question counts as a hit at *k*
if any chunk from an expected file appears in the top *k*.

Reproduce: `uv run python eval/run_eval.py --label <name> --strategy <s> --variant <v> --provider <p>`

## Runs

| run | chunking | embedding model | chunks | dup % | over limit % | r@1 | r@5 | r@10 | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-text-minilm` | text (1000c/200o) | MiniLM-L6-v2 (256 tok) | 2976 | 12.6 | **73.2** | 0.133 | 0.500 | 0.600 | 0.278 |
| `ast-minilm` | AST everywhere | MiniLM-L6-v2 (256 tok) | 7833 | 4.9 | 0.0 | 0.200 | 0.467 | 0.567 | 0.312 |
| `text-openai` | text (1000c/200o) | text-embedding-3-small | 2976 | 12.6 | 0.0 | 0.300 | 0.567 | **0.700** | **0.436** |
| `ast-openai` | AST, budget = 8191 | text-embedding-3-small | 2036 | 0.5 | 0.0 | 0.200 | 0.500 | 0.567 | 0.304 |
| `ast-openai-tuned` | AST, tuned budgets | text-embedding-3-small | 2597 | 0.8 | 0.0 | 0.200 | 0.433 | 0.600 | 0.313 |
| **`astcode-openai`** | **AST for code only** | text-embedding-3-small | 2440 | 1.9 | 0.0 | **0.333** | 0.567 | 0.667 | 0.433 |

Per-segment MRR:

| run | pinpoint | behavioral | architectural |
|---|---:|---:|---:|
| `baseline-text-minilm` | 0.400 | 0.215 | 0.161 |
| `text-openai` | 0.525 | **0.499** | 0.130 |
| **`astcode-openai`** | **0.633** | 0.396 | 0.107 |

## Findings

**1. The baseline was silently truncating 73% of its chunks.**
`all-MiniLM-L6-v2` accepts 256 tokens; a 1000-character chunk of real code
tokenizes to ~271. Most chunks lost their tail before ever being embedded. This
was invisible until measured.

**2. Naive text chunking produced 12.6% exact-duplicate chunks.**
FastAPI's `routing.py` repeats identical `Doc(...)` docstring boilerplate, and
character-window splitting sliced it into byte-identical chunks. One observed
query filled 4 of its 5 context slots with the same text — effectively 2
distinct sources instead of 5. Structural chunking cut this to 1.9%.

**3. The embedding model was the single biggest lever.**
Swapping MiniLM → `text-embedding-3-small` with chunking held constant moved
MRR 0.278 → 0.436 (+57%). Most of the "chunking problem" was a context-window
problem wearing a disguise.

**4. A model's token limit is a ceiling, not a target chunk size.**
The first AST run passed the model's full 8191-token limit down as the chunk
budget. Chunks ballooned (largest 8186 tokens) and MRR *fell* to 0.304 — a long
passage averages into a vector that matches nothing specifically. Separating
`TARGET_PROSE_TOKENS=400` and `MAX_CODE_CHUNK_TOKENS=1200` from the hard limit
fixed it.

**5. "AST everywhere" looked like a regression; isolating the variable showed the opposite.**
`ast-openai-tuned` scored *below* `text-openai` (0.313 vs 0.436), which reads as
"AST chunking doesn't work." But the corpus is 144 Markdown files against 42
Python ones, so that run had also replaced the *prose* chunker — and lost the
baseline's 200-character overlap. Re-running with AST applied to code only
(`astcode-openai`) isolated the effect:

- `pinpoint` MRR **0.525 → 0.633** (+21%), r@1 **0.333 → 0.500** (+50%)
- overall MRR essentially flat (0.436 → 0.433)

So structural chunking is a clear win *for code questions*, and the custom
Markdown chunker is a regression that the combined run had hidden. Per-language
chunking is the right conclusion, not "abandon AST."

**6. Architectural questions are unsolved by chunking alone.**
r@1 is **0.000** for architectural questions in *every* run, and MRR got slightly
worse as code retrieval improved (0.161 → 0.107). Questions like "how does a
request flow from ASGI entry to endpoint?" have no single chunk that answers
them. This is the motivating evidence for step 5 (file-level cards).

## Chosen configuration

`ast-code` chunking + `text-embedding-3-small`: best r@1 (0.333) and by far the
best `pinpoint` MRR (0.633), with `text-openai`-equivalent overall MRR.

## Open items carried into later steps

- Markdown chunker underperforms LangChain's splitter — needs overlap and better
  section handling (currently prose deliberately stays on the baseline splitter)
- 6 questions still miss entirely at k=20: `b05`, `b06`, `b07`, `b12`, `a03`, `a06`
- Architectural retrieval needs file cards (step 5)
- `pinpoint` questions should be far better served by lexical symbol search (step 4)
- Untested: does indexing `tests/` help or hurt? The harness makes this measurable
