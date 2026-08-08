# Retrieval results — chunking (Phase A) and hybrid search (Phase B)

**Corpus:** `fastapi` pinned at tag `0.115.0` (SHA `40e33e49`), restricted to
`fastapi/` + `docs/en/` — 201 files (42 Python, 144 Markdown).

**Golden set:** 54 hand-labelled questions in `golden.jsonl`, tagged `pinpoint`
(12), `behavioral` (12), `architectural` (15), `paraphrase` (15). A question
counts as a hit at *k* if any chunk from an expected file appears in the top *k*.

> The set grew twice, and both times it changed conclusions:
> **30 → 38** added `paraphrase` questions after finding the original set was
> biased toward lexical search (finding B2). **38 → 54** doubled the two weakest
> segments after realising 6 architectural questions could not support a
> conclusion (finding B5).
>
> The Phase A table below is from the original 30-question runs and is kept for
> the *relative* chunking comparison only. The Phase B table is the definitive
> one, on all 54.

Reproduce: `uv run python eval/run_eval.py --label <name> --strategy <s> --variant <v> --provider <p>`

## Runs (30-question set — relative comparison only)

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

## Phase A chosen configuration

`ast-code` chunking + `text-embedding-3-small`: best r@1 and by far the best
`pinpoint` MRR, with `text-openai`-equivalent overall MRR.

---

# Phase B — step 4: hybrid retrieval with rank fusion

Added a SQLite FTS5 lexical index (BM25, column weights: symbol 5, path 2,
content 1) alongside the vector store, merged with Reciprocal Rank Fusion.
Both indexes key chunks by the same deterministic id (`pipeline/ids.py`) so the
two ranked lists can be fused.

All runs below use `ast-code` chunking + `text-embedding-3-small`, on all 54
questions.

| run | mode | r@1 | r@5 | r@10 | MRR | pinpoint | behavioral | architectural | paraphrase |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline-text-minilm` | semantic | 0.185 | 0.519 | 0.685 | 0.334 | 0.400 | 0.215 | 0.218 | 0.492 |
| `astcode-openai` | semantic | 0.333 | 0.593 | 0.722 | 0.452 | 0.633 | 0.396 | 0.307 | **0.499** |
| `astcode-openai-lexical` | lexical | **0.426** | 0.685 | 0.759 | **0.530** | **0.792** | **0.738** | 0.308 | 0.375 |
| **`astcode-openai-hybrid`** | hybrid 1:2, k=60 | 0.333 | **0.778** | **0.815** | 0.507 | 0.708 | 0.520 | **0.389** | 0.454 |

Full baseline → chosen hybrid: r@5 **0.519 → 0.778** (+50%), r@10 **0.685 →
0.815**, MRR **0.334 → 0.507** (+52%), `pinpoint` r@5 **0.750 → 1.000**,
`architectural` MRR **0.218 → 0.389** (+78%).

Weight sweep: `uv run python eval/sweep.py --variant astcode-openai --provider openai`

## Findings

**B1. Lexical search beats semantic search on identifier-bearing questions.**
BM25 alone reached MRR 0.530 against 0.452 for embeddings, and on `pinpoint`
questions 0.792 against 0.633. Embeddings place `include_router` in almost the
same neighbourhood as every other router method; BM25 does not.

**B2. The original golden set was biased toward lexical — finding that mattered more than any number here.**
The first 30 questions were written while reading the source, so they contained
exact symbol names, which is exactly what BM25 is best at. Real users often don't
know the identifier — that is *why* they are asking. Adding `paraphrase` questions
in user vocabulary ("how do I let someone upload a picture?" rather than "where is
`UploadFile`?") inverts the ranking:

| | pinpoint MRR | paraphrase MRR | paraphrase r@5 |
|---|---:|---:|---:|
| semantic | 0.633 | **0.499** | 0.733 |
| lexical | **0.792** | 0.375 | 0.400 |
| hybrid | 0.708 | 0.454 | **0.800** |

Lexical `paraphrase` r@5 is **0.400 against 0.733** for semantic — it does not
just lose, it fails. That is the real case for fusion: not that hybrid wins every
segment, but that it never collapses on either query shape, and its `paraphrase`
r@5 (0.800) beats both inputs.

**B3. Naive 1:1 fusion is worse than either input.**
Equal weighting scored MRR 0.470, below both lexical (0.530) and the best hybrid.
It dilutes the stronger ranking with weaker candidates. A hybrid retriever shipped
with default weights would have looked like proof that "hybrid doesn't help."

**B4. The metric you optimise should match how the system is used.**
Pure lexical has the better overall MRR (0.530 vs 0.507) and r@1 (0.426 vs 0.333),
so an MRR-maximising sweep picks lexical or a lexical-heavy hybrid. But the
generator is handed ~6 chunks, so what matters is whether the right file is in the
context *at all* — recall@5, not rank 1. Optimising r@5 selects different weights
(1:2 at k=60) and yields **0.778 vs 0.685**, a four-question difference, while
also giving the best `architectural` MRR and best `paraphrase` r@5.

**B5. Two of the four segments were too small to support conclusions, and the small samples were badly wrong.**
The Phase A/early-B runs measured `architectural` on 6 questions. Doubling that
segment to 15 moved the semantic estimate from **0.107 to 0.307** — nearly 3×.
The earlier number was not a mild underestimate, it was noise.

| segment MRR (semantic) | n=6/8 estimate | n=15 measured |
|---|---:|---:|
| architectural | 0.107 | **0.307** |
| paraphrase | 0.431 | **0.499** |

Had step 5 (file cards) been built against the 6-question figure, it would have
targeted a mis-measured problem and any "improvement" would have been
unattributable. Growing the set before building on it was the correct order.

**B6. Architectural retrieval is weak but not catastrophic.**
On 15 questions the best `architectural` MRR is 0.389 (hybrid) with r@5 0.533 —
still the weakest segment, and still the target for step 5, but the gap to the
other segments is far smaller than the 6-question sample implied. Hybrid already
improved it from 0.218 (baseline) to 0.389.

## Chosen configuration

`mode=hybrid`, `semantic_weight=1.0`, `lexical_weight=2.0`, `rrf_k=60`,
`candidate_k=40` — set as defaults in `pipeline/config.py`. Chosen on recall@5
rather than MRR, for the reason in B4.

---

# Phase B — step 5: structural cards + result diversity

Two changes, measured as a 2×2 so their effects are separable:

1. **Structural cards** (`pipeline/cards.py`) — extra chunks that describe
   structure rather than contain code. A `file_card` per module (path, docstring,
   classes with methods, functions, imports) and a `package_card` per directory
   (its modules and their main symbols). Built from the AST, not an LLM: 191 cards
   for 201 files, at zero API cost and no measurable indexing time.
2. **Result diversity** (`max_per_file=2`) — cap chunks per file in the returned
   set. Question `a02` had *four of its top five slots* filled by the same docs
   page, so the generator saw one source instead of five.

| cards | diversity | r@1 | r@5 | r@10 | MRR | arch r@5 | arch r@10 | arch MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no | no | 0.333 | 0.778 | 0.815 | 0.507 | 0.533 | 0.600 | 0.389 |
| yes | no | 0.370 | 0.778 | 0.815 | 0.531 | 0.600 | 0.600 | 0.372 |
| no | yes | 0.333 | 0.778 | 0.870 | 0.516 | 0.533 | 0.733 | 0.400 |
| **yes** | **yes** | **0.370** | **0.796** | **0.889** | **0.543** | **0.600** | **0.800** | 0.386 |

Per segment, shipped configuration vs step 4:

| segment | step 4 MRR | step 5 MRR | step 4 r@10 | step 5 r@10 |
|---|---:|---:|---:|---:|
| overall | 0.507 | **0.543** | 0.815 | **0.889** |
| architectural | 0.389 | 0.386 | 0.600 | **0.800** |
| behavioral | 0.520 | **0.574** | 0.917 | 0.917 |
| paraphrase | 0.454 | **0.527** | 0.800 | **0.867** |
| pinpoint | 0.708 | **0.729** | 1.000 | 1.000 |

## Findings

**B7. The metric could not credit the feature being built, and fixing that came first.**
Package cards have directory paths (`fastapi/middleware/`), which never equal an
expected file like `fastapi/middleware/cors.py`. On the first run the middleware
package card ranked **4th** for `a09` and was scored as a miss, while the "first
hit" was recorded at rank 9. Retrieving that card *is* a correct answer to "what
middleware ships with FastAPI." `first_hit_rank` now credits a directory-shaped
path when an expected file lives under it, and **both** arms were re-measured
under the corrected metric — the table above is post-fix throughout. Without this
the conclusion would have been "cards make architectural retrieval worse."

**B8. Cards improve recall, not rank.**
`architectural` r@10 went **0.600 → 0.800** (three more questions of fifteen) while
`architectural` MRR stayed flat (0.389 → 0.386). Cards help the right file appear
*at all*; they do not push it to the top. For a RAG generator that reads ~6 chunks
this is the useful direction, but it should not be described as "fixing ranking."

**B9. Diversity capping was the larger win for architectural questions, and it was free.**
`max_per_file=2` alone lifted `architectural` r@10 from 0.600 to 0.733 with no new
chunks, no indexing cost, and about ten lines of code — comparable to what the
entire card subsystem delivered. Architectural questions expect *several* files,
so spending five slots on one file is the exact wrong allocation.

**B10. Package cards are mostly not retrieved; file cards carry the benefit.**
Across 54 questions a package card reached the top 5 only twice (`p07` →
`fastapi/openapi/`, `a09` → `fastapi/middleware/`). The per-file cards do the work.
Package cards are cheap enough to keep, but they are not the mechanism.

**B11. Cards cost a little precision on pinpoint questions.**
`p01`, `p03` and `p05` each slipped exactly one rank as cards displaced code
chunks, and `a12` fell from rank 1 to 5. Net effect is still clearly positive
(15 questions improved, 5 regressed), but cards are not free: they add candidates
that compete with the real definitions.

## Chosen configuration

Cards on (`INDEX_CARDS=True`), `MAX_CHUNKS_PER_FILE=2`, plus the step 4 hybrid
settings. Reproduce step 4 with `index_repo.py --no-cards` and
`run_eval.py` without `--max-per-file`.

---

# Phase B — step 5b: stratified retrieval (fixing docs crowding)

## The defect

Documentation crowded source code out of the results for code questions. `a03`
("how is the security system organized?") returned six Markdown chunks and no
`fastapi/security/*.py` at all. Noted as far back as step 1 (top-10 was 9 Markdown
/ 1 Python) and still unfixed after step 5.

The cause was not ranking quality but **corpus imbalance at the chunk level**:

| | chunks | share |
|---|---:|---:|
| prose (markdown, yaml, html, css) | 2201 | **83.7%** |
| code (python) | 430 | 16.3% |

At a 5:1 ratio, prose wins the top-k on volume regardless of relevance.

## The fix

Give code and prose **separate rank spaces** and fuse them: four ranked lists
({semantic, lexical} × {code, prose}) instead of two. Because RRF scores by
position within a list, the best code chunk gets rank-1 treatment even when it
would place 30th in a pooled list. Candidates are fetched *per stratum* from the
stores (Chroma `$in`/`$nin`, a new `language` column in FTS5) — filtering one
pooled list would not work, since a pool of 40 can contain almost no code.

| config | r@1 | r@5 | r@10 | MRR | pinpoint | behavioral | architectural | paraphrase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step 5 (pooled) | 0.370 | 0.796 | 0.889 | 0.543 | 0.729 | 0.574 | 0.386 | **0.527** |
| code floor = 2 | 0.389 | 0.833 | 0.926 | 0.557 | 0.729 | 0.560 | 0.469 | 0.505 |
| code floor = 5 | 0.389 | 0.870 | **0.963** | 0.568 | 0.729 | 0.560 | 0.550 | 0.465 |
| **stratified** | **0.574** | **0.870** | 0.926 | **0.706** | **0.875** | **0.785** | **0.761** | 0.452 |

`architectural` goes from MRR 0.386 to **0.761** with r@5 and r@10 both at
**1.000**. Overall MRR rises 0.543 → 0.706, the largest single improvement in the
project.

## Findings

**B12. A softer intervention was tried first and was clearly worse.**
A "code floor" (pooled fusion, then top up if fewer than N code results made the
cut) reached only MRR 0.568 against stratification's 0.706. The reason is
positional: the floor injects code *after* fusion, so it enters at the bottom of
the list, whereas stratification lets code compete from rank 1 of its own space.
Where an intervention happens in the pipeline mattered more than how much code it
forced in.

**B13. Equal stratum weighting is the only stable point — it is not a tunable.**
Tilting toward prose collapses the system, and the boundary is at exactly parity:

| prose_weight (code = 1.0) | overall MRR |
|---|---:|
| 1.0 | **0.706** |
| 1.1 | 0.349 |
| 1.2 | 0.312 |
| 1.5 | 0.258 |

Any tilt makes one stratum systematically outrank the other at equal
within-stratum rank; since each stratum supplies more candidates (40) than there
are slots (20), the lighter stratum is shut out almost entirely. Noted in
`config.py` so nobody "optimises" it later.

**B14. The fix has a real cost: `paraphrase` regressed.**
`paraphrase` MRR fell 0.527 → 0.452 and r@5 0.800 → 0.600, and `n12`/`n13` now
miss entirely. Those questions are answered by documentation, so guaranteeing code
half the slots displaces the pages that answer them. The trade is 0.075
`paraphrase` MRR for 0.375 `architectural` MRR and 0.163 overall — worth taking,
but it is a trade, not a free win. The principled fix is query-dependent
stratum weighting (identifier-bearing queries favour code, prose-shaped queries
favour docs), deliberately not attempted here to avoid overfitting 54 questions.

## Chosen configuration

`STRATIFY_RETRIEVAL=True`, `CODE_WEIGHT=PROSE_WEIGHT=1.0`, plus step 4/5
settings. Reproduce the pooled behaviour by omitting `--stratify`.

## Open items carried into later steps

- **`paraphrase` regression** from stratification (see B14) — query-dependent
  stratum weighting is the proposed fix
- 3 questions miss entirely at k=20: `n03`, `n12`, `n13` — all `paraphrase`,
  consistent with B14 rather than a separate defect
- `behavioral`/`pinpoint` remain n=12; grow before trusting small differences
- Untested: does indexing `tests/` help or hurt?
- Untested: `jina-embeddings-v2-base-code` vs `text-embedding-3-small`
- **Behavioral MRR regressed** under the r@5-tuned weights (lexical 0.738 →
  hybrid 0.520) even though its r@5 improved (0.917 → 0.833). Worth a per-segment
  weighting experiment rather than one global setting
- **Markdown chunker** underperforms LangChain's splitter; prose deliberately
  stays on the baseline splitter for now
- 4 questions still miss entirely at k=20 under hybrid: `b06`, `a06`, `n03`, `a08`
- `behavioral` and `pinpoint` are still n=12; at 54 questions one question moves
  overall MRR by ~0.019, so differences under ~0.04 should not be treated as real
- Untested: does indexing `tests/` help or hurt? The harness makes it measurable
- Untested: a code-specific embedding model (`jina-embeddings-v2-base-code`)
  against `text-embedding-3-small`, now that the harness can settle it
