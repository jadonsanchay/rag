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

---

# Step 6 — grounded answers, citations, and refusal

Retrieval metrics say whether the right file was found. They say nothing about
whether the answer is grounded in it, whether its citations are real, or whether
the system admits ignorance. `eval/answer_eval.py` grades those.

The golden set gained **8 `unanswerable` questions** — refusal cannot be measured
without a negative class. Their subjects were chosen by grepping the corpus for
zero matches (`billing`, `tenant`, `feature flag`, `circuit breaker`, `LDAP`,
`audit log`, `AWS region`, `users table`), because several obvious candidates were
contaminated: `redis` appears in 5 files, `kubernetes` 5, `rust` 12, `ORM` 163.
`run_eval.py` excludes them from retrieval metrics, since a question with no
correct file would otherwise score as a permanent recall miss.

| metric | initial prompt | calibrated prompt |
|---|---:|---:|
| refusal rate on answerable | 0.259 | **0.074** |
| — unwarranted (generator's fault) | 0.148 | **0.019** |
| — warranted (retrieval missed) | 0.111 | 0.056 |
| citation rate | 0.741 | **0.926** |
| grounded rate (cites an expected file) | 0.593 | **0.815** |
| **refusal rate on unanswerable** | 1.000 | **1.000** |
| **hallucination rate on unanswerable** | 0.000 | **0.000** |
| citations verified | 66/66 | **131/131** |
| fabricated source numbers | 0 | **0** |

Per segment (calibrated):

| segment | n | false refusal | grounded |
|---|---:|---:|---:|
| pinpoint | 12 | 0.000 | **1.000** |
| behavioral | 12 | 0.000 | 0.917 |
| architectural | 15 | 0.000 | 0.933 |
| paraphrase | 15 | 0.267 | 0.467 |

## Findings

**B15. The refusal metric was blaming the wrong component.**
`false_refusal_rate` counted every refusal of an answerable question. But 6 of the
first 14 refusals happened when retrieval had *not* surfaced the expected file —
the model had nothing to work with, and refusing was correct. Splitting the metric
into **unwarranted** (right file was in context, refused anyway — the generator's
bug) and **warranted** (context lacked the answer — upstream) moved the generator's
true failure rate from 0.259 to 0.148 before any code changed. A single number was
holding two different failures.

**B16. A prompt phrase created a refusal attractor and broke everything.**
The first attempt at softening refusal included "…answer what they collectively
support, naming any part you could not determine." That phrase mirrors the refusal
template ("one sentence naming what is missing"), and the model pattern-matched
into it: **refusal rate went to 1.000 and citation rate to 0.000**, refusing even
"where is the `APIRouter` class defined?" with the file path visibly in its
context. Rewriting to state that each source's `path:lines` label is authoritative,
and gating refusal behind "only when the sources are entirely unrelated", took
refusal to 0.074. Prompt wording near a sentinel behaves like an attractor, and the
eval caught in one run what spot-checking would have missed.

**B17. More context made refusal worse, so it was never slot starvation.**
The obvious hypothesis was that stratification left prose only 3 of 6 slots. Raising
`top_k` from 6 to 10 moved refusal the wrong way (0.259 → 0.315) and `paraphrase`
from 0.667 to 0.733. Extra context added noise, not evidence. Worth recording as a
rejected hypothesis, because it is the change most people would try first.

**B18. Recall improved without trading away refusal integrity.**
The usual failure mode when reducing over-refusal is that the model starts
answering things it shouldn't. It did not: `unanswerable` refusal held at **1.000**
and hallucination at **0.000** across both prompts. That is the property worth
guarding, and it is only checkable because the negative class exists.

**B19. Citation verification found nothing wrong — which is itself the result.**
All 131 cited spans exist on disk with valid line ranges, and no answer ever cited
a source number it wasn't given. The check is mechanical (no LLM), so it costs
nothing to keep in the request path, and `ask.py` marks each source `OK` or `!!`.
A clean result here is only meaningful because the checker is known to work: it
caught the class-skeleton span bug in step 5 (`routing.py:593-4437`).

**B20. `paraphrase` remains the weakest segment at every layer.**
Grounded rate 0.467 and false refusal 0.267, against ≥0.917 grounded for the other
three. This traces directly to the step 5b stratification trade (B14): guaranteeing
code half the slots displaces the documentation those questions need. The answer
layer makes the cost concrete — a 0.2 drop in `paraphrase` r@5 became a 0.267 false
refusal rate.

---

# Step 6b — LLM judge for claim-level faithfulness

`citations.py` proves a cited span exists; it cannot tell whether the claim matches
what is at that span. An answer could cite `routing.py:593-618` and assert
"APIRouter manages database connections" and pass every mechanical check. Closing
that gap needs a model.

Two design decisions:

- The judge grades **faithfulness to the provided sources**, not correctness in the
  abstract. Asking "is this true about FastAPI?" invites the judge to answer from
  its own knowledge of a very popular library, which measures the judge instead of
  the system.
- The judge is a **different, stronger model than the generator** (`gpt-4o` judging
  `gpt-4o-mini`). A model grading its own output shares its blind spots.

## The judge was validated before it was trusted

`uv run python eval/run_judge.py --validate` injects a claim no source can support
("controlled by the `FASTAPI_STRICT_MODE` environment variable", "memoised in a
Redis cache", "delegates to the billing module") into real answers and checks the
judge flags it.

| validation attempt | detection |
|---|---:|
| first 8 answers (all `pinpoint`) | 8/8 = 1.000 |
| stratified across all 4 types, comparing unsupported counts | 10/12 = 0.833 |
| stratified, checking the injected claim specifically | **11/12 = 0.917** |

Both refinements mattered:

**The easy sample overstated the judge.** `[:8]` returned only `pinpoint` answers,
which are one or two claims long. Detecting an injection there is trivial compared
with a 13-claim architectural answer. Stratifying the sample dropped the measured
detection rate from 1.000 to 0.833.

**The validation method was itself confounded.** Comparing unsupported-claim counts
before and after injection assumes stable claim decomposition, and the judge's
decomposition is not stable: the same answer decomposed into 12 claims on one run
and 13 on another, and `a01`'s baseline unsupported count moved 4 → 2 on identical
input at `temperature=0`. Checking whether the judge extracted *the injected claim
specifically* is immune to that, and raised the measured rate to 0.917. The single
remaining failure was a `DROPPED` case — the judge never extracted the claim at
all, rather than judging it wrong.

**Consequence for reading the numbers below:** the judge has real run-to-run
variance. Aggregate faithfulness differences under roughly 0.02 are noise, and
per-answer verdicts are stronger evidence than aggregate deltas.

## Results

| metric | before class-context fix | after |
|---|---:|---:|
| answers graded | 50 | 50 |
| total claims | 282 | 303 |
| claim faithfulness | 0.972 | 0.964 |
| mean answer faithfulness | 0.984 | 0.972 |
| fully clean answers | 0.920 | 0.860 |
| **contradicted claims** | **0** | **0** |

## Findings

**B21. The judge found a chunking defect that no retrieval metric could see.**
For `a12` ("how is dependency metadata represented internally?") the model claimed
"`Security` is a subclass of `Depends`" — which is *true* (`params.py:773`). The
judge marked it unsupported. Checking the retrieved chunks showed why: they started
at `params.py:761` and `params.py:774`, the *method bodies*. `class Depends:` is at
line 760 and `class Security(Depends):` at 773, so the declarations carrying the
inheritance were never in the context. The model was reciting prior knowledge.

Retrieval metrics were blind to this: `params.py` *was* retrieved, so file-level
recall scored a perfect hit. Only claim-level grading exposed it.

The fix prepends the class declaration line to every method chunk (tracked with
`header_lines`, so citation verification still maps text back to real lines).
Verified per-answer:

| question | faithfulness before | after |
|---|---:|---:|
| a12 | 0.727 | **1.000** |
| a10 | 0.900 | **1.000** |
| n12 | 0.800 | **1.000** |

The claim now reads `[supported] The Security class extends Depends`. Retrieval r@5
also rose 0.870 → 0.889 and `behavioral` r@5 0.917 → 1.000, at the cost of some
MRR (0.706 → 0.663) — acceptable under the recall@5 rationale in B4.

**B22. The aggregate did not move; the specific defect did.**
Claim faithfulness went 0.972 → 0.964 while the three targeted answers went to
1.000 and new answers (`a08`, `a14`, `n04`, `n02`) appeared in the worst list. Given
the judge variance documented above, the aggregate change is not interpretable as a
regression. Stated plainly rather than presented as an improvement: the fix is
verified at the level of the defect it targeted, not at the level of the summary
statistic.

**B23. The residual unfaithfulness is concentrated in synthesis and invented examples.**
Of the flagged claims, the recurring shapes were narrative bridging on architectural
questions ("After the endpoint function completes, it returns a response…" — plausible
filler spanning a gap between chunks) and fabricated code examples on `paraphrase`
questions (`n12` invented an `@app.post("/items/")` snippet). Zero claims were ever
marked *contradicted* across both runs, so the failure mode is unsupported
elaboration, not stating things the sources refute.

## Open items

- **`paraphrase` end-to-end quality** remains the top defect; cause known
  (B14/B20/B23). Query-dependent stratum weighting is the fix, not attempted
  because tuning a query classifier against 15 questions would overfit
- **Answer correctness is still unmeasured.** Faithfulness asks whether the sources
  support each claim, not whether the answer is right. A grounded answer built from
  a wrong-but-retrieved chunk still scores 1.000. Reference answers or human
  grading would be needed
- **Judge variance is unquantified.** It was observed, not measured. Running the
  judge N times per answer and reporting a confidence interval would be the
  correct treatment
- 1 unwarranted refusal remains (1 of 54)
- `behavioral`/`pinpoint` remain n=12; grow before trusting small differences
- Judging costs roughly $0.75 per full pass with `gpt-4o`
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

---

# Step 12 — query rewriting for multi-turn

A follow-up like "and where is that called from?" is unretrievable as written:
embedded alone it matches nothing in particular, and BM25 sees only stopwords. The
referent lives in the prior turns, so the question is condensed into a standalone
query *before* it reaches either index.

Measured on 8 two-turn exchanges against the fastapi corpus
(`uv run python eval/rewrite_eval.py`):

| | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|
| raw follow-up | 0.375 | 0.625 | 0.259 |
| **rewritten** | **1.000** | **1.000** | **0.823** |

Three of eight went from *not found in the top 10* to rank 1:

| follow-up | rewritten to | rank |
|---|---|---|
| "where is that class defined?" | "Where is the BackgroundTasks class defined?" | miss → 1 |
| "which module holds it?" | "Which module holds the compatibility layer for Pydantic v1 and v2?" | miss → 1 |
| "what does it use to do that?" | "What does FastAPI use to run an endpoint in a threadpool?" | miss → 4 |

## Findings

**B24. Rewriting is the single highest-leverage mechanism measured in this project.**
MRR more than tripled (0.259 → 0.823) from one cheap call to a small model. No
retrieval change in Phase B came close to that on its own.

**B25. A cheap pre-filter keeps it from costing anything on normal questions.**
Rewriting only runs when the question looks dependent — a follow-up opener, a
pronoun, or fewer than five words. A self-contained question skips the call
entirely, so the latency and cost land only where they buy something.

**B26. The rewritten query is shown in the UI, not hidden.**
Retrieval acting on a query the user did not type is the most confusing thing this
system can do. The `rewrite` SSE frame carries the condensed query and the
transcript renders it ("searched for: …"), so a surprising result is explainable
rather than mysterious.

## Limitation

n=8, hand-written by the same person who wrote the rewriter's prompt. The effect
size is large enough that the direction is not in doubt, but the absolute numbers
should not be quoted as precise. Rewrite quality also depends on the prior turn
being correct: a wrong first answer poisons the referent for the follow-up.
