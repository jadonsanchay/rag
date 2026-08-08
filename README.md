# Codebase Q&A

Ask questions about a real codebase and get answers with verifiable `file:line`
citations. Retrieval is hybrid (BM25 + embeddings, fused by rank), chunking is
AST-aware, and every change to the pipeline was justified by a measured number
rather than a guess — see **[eval/RESULTS.md](eval/RESULTS.md)**.

```
$ uv run python ask.py "How does FastAPI decide to run my endpoint in a threadpool?"

FastAPI runs the endpoint in a threadpool when the path operation function is a
normal `def` rather than `async def` [2]. ...

Sources:
 OK [2] fastapi/routing.py:204-214
    [3] docs/en/docs/async.md
Citations: 3/3 citations verified
```

## Why this exists

Most RAG demos stop at "it returns something plausible." The interesting problems
turn out to be elsewhere:

- **Chunking beats model choice.** Naive text splitting produced 12.6% byte-identical
  duplicate chunks, and the embedding model was silently truncating 73% of them.
- **Embeddings are bad at identifiers.** `include_router` sits in nearly the same
  vector neighbourhood as every other router method. BM25 is not confused by this,
  which is why retrieval is hybrid.
- **Documentation drowns source code.** Prose outnumbered code 5:1 at the chunk
  level, so docs won the top-k on volume. Fixed by giving code and prose separate
  rank spaces.
- **An answer you cannot check is not useful.** Citations are verified against the
  working tree, and the system refuses rather than guessing.
- **A follow-up is unretrievable as written.** "And where is that called from?"
  embeds to noise. Condensing it against the history first moved retrieval MRR
  from 0.259 to 0.823.
- **"Supports language X" is easy to claim and easy to get wrong.** The first
  TypeScript spec recognised classes and `function` declarations — and matched
  almost nothing in a real Next.js project, where every component is
  `const Foo = () => {}`.

Headline movement on a 62-question labelled set: retrieval MRR **0.334 → 0.706**,
recall@5 **0.519 → 0.889**, with `pinpoint` and `architectural` both at recall@5 =
**1.000**. Refusal on unanswerable questions: **8/8**, hallucination rate **0.000**.

## Setup

```bash
uv sync
cp .env.example .env      # then add OPENAI_API_KEY
```

Clone a repository to index (kept outside the project tree — a vendored
`pyproject.toml` inside it makes uv treat the clone as a package source):

```bash
mkdir -p ~/.cache/codebase-qa/repos
git clone --depth 1 --branch 0.115.0 \
  https://github.com/fastapi/fastapi.git ~/.cache/codebase-qa/repos/fastapi
```

## Usage

**Index** — walk, chunk, embed, and build both indexes:

```bash
uv run python index_repo.py ~/.cache/codebase-qa/repos/fastapi \
  --include fastapi --include docs/en --variant astcode-cards
```

`--include` restricts the corpus to named subtrees, which keeps an eval corpus
reproducible. FastAPI ships 26 translations of its docs and 701 near-duplicate
tutorial snippets; indexing them floods the index with near-identical chunks.

Any repo works, not just the sample. The manifest records where each indexed tree
lives, so a repo outside the cache directory still resolves for citation
verification and the source viewer:

```bash
uv run python index_repo.py ~/code/my-typescript-app --variant ts
uv run python ask.py "How is the websocket server set up?" --repo my-typescript-app --variant ts
```

## Languages

Structural (declaration-aware) chunking: **Python, TypeScript, JavaScript, Go,
Rust, Java** — plus TSX/JSX. Everything else falls back to line-aware text
windows, which still carry real line numbers for citations.

Python uses the stdlib `ast` module; the rest use tree-sitter. That split is
deliberate rather than lazy: `ast` understands docstrings and decorators natively
and its output was already measured, so the `Chunker` interface exists to let the
better parser win per language rather than to force one code path. Adding a
language is a `LanguageSpec` entry (which node types are functions, which are
containers) plus a smoke test — no grammar queries to write.

**Search** (retrieval only, with the trace showing which retriever found what):

```bash
uv run python query.py "Where is solve_dependencies implemented?"
# [1] score=0.3561  fastapi/dependencies/utils.py:562-685  [solve_dependencies]
#       via lexical:code#1, semantic:code#15
```

**Ask** (retrieval + generation + citation verification):

```bash
uv run python ask.py "How is the security system organized?"
```

**Serve** an HTTP API:

```bash
uv run uvicorn api.main:app --reload --port 8000
# http://localhost:8000/docs
```

| endpoint | purpose |
|---|---|
| `POST /repos` | add a public GitHub URL; clones and indexes in the background (202) |
| `GET /repos/{id}` | poll the indexing stage: `queued → cloning → indexing → ready` |
| `POST /repos/{id}/reindex` | re-clone and rebuild |
| `DELETE /repos/{id}` | drop indexes, registry row, and the clone |
| `POST /conversations` | start a thread against one repo |
| `POST /conversations/{id}/messages` | SSE: `rewrite` → `trace` → `token`… → `done` |
| `POST /ask` | stateless single question; SSE `trace` → `token`… → `done` |
| `GET /repos` | indexed collections and their manifests |
| `GET /file` | a slice of a source file, for the viewer behind a citation |
| `GET /health` | liveness |

`/ask` emits the retrieval trace **before** generation starts. Measured warm:
trace at ~640ms, first token ~1.6s, complete ~3.0s — so the UI can render sources
and let you start reading code roughly two seconds before the answer finishes.
Retrieval and generation both block, so they are pumped on a worker thread;
concurrent requests return their traces within 1ms of each other rather than
serialising.

**Web UI** (needs the API running on port 8000):

```bash
cd web
npm install
npm run dev            # http://localhost:5173
```

Paste a GitHub URL to index a repo, switch between indexed repos, then chat. Each
turn shows the streamed answer with clickable `[n]` citation markers, the sources
with the trace explaining why each ranked (`lexical:code #3 · semantic:code #1`),
and — for follow-ups — the standalone query the question was rewritten into. Click
any source to open the file at the cited lines.

`EventSource` cannot issue a POST, so SSE frames are parsed by hand off the fetch
body stream (`web/src/api.ts`). The answer renders with a regex walk rather than a
markdown parser: a half-streamed answer is frequently invalid markdown — unclosed
fences, dangling backticks — and a parser flickers as tokens arrive.

```bash
npx playwright install chromium
npm run smoke          # 11 browser checks
```

The smoke test asserts the thing unit tests cannot: that sources are on screen
*while the answer element does not yet exist*, that citation spans verify, that
clicking a marker highlights its source, and that an unanswerable question is
refused rather than answered.

## Evaluation

The part that makes the rest trustworthy. Nothing was shipped on a hunch.

```bash
uv run python eval/run_eval.py --label mine --strategy ast-code \
  --variant astcode-cards --provider openai --mode hybrid \
  --semantic-weight 1.0 --lexical-weight 2.0 --rrf-k 60 --max-per-file 2 --stratify

uv run python eval/compare.py baseline-text-minilm mine   # per-question diff
uv run python eval/sweep.py                               # fusion weight sweep
uv run python eval/answer_eval.py --label mine            # refusal + citation validity
uv run python eval/run_judge.py --validate                # is the judge trustworthy?
uv run python eval/run_judge.py --label mine              # claim-level faithfulness
```

62 hand-labelled questions in `eval/golden.jsonl`, tagged `pinpoint`, `behavioral`,
`architectural`, `paraphrase`, and `unanswerable`. The set grew twice, and both
times it overturned a conclusion — the original questions were biased toward
lexical search, and a 6-question segment gave an estimate that was 3× off. The
`unanswerable` class exists because refusal cannot be measured without one.

The LLM judge is validated by perturbation before it is trusted: inject a claim no
source can support, and confirm the judge flags it (**11/12**). Validating it
against only the easy questions had reported 8/8.

## Layout

```
pipeline/          library code
  repo_loader.py     walk a repo, ignore rules, language detection
  chunkers/          stdlib-ast (Python), tree-sitter (5 languages), text fallback
  cards.py           file/package structure summaries
  embeddings.py      swappable providers, exposing their token limits
  vector_store.py    ChromaDB
  lexical_index.py   SQLite FTS5 + BM25
  fusion.py          reciprocal rank fusion
  retriever.py       hybrid + stratified retrieval
  generator.py       cited answers, refusal contract, streaming
  citations.py       verify cited spans against the working tree
  manifests.py       where each indexed repo actually lives on disk
  registry.py        SQLite: indexed repos, their status, conversations
  ingest_job.py      clone + background indexing with guards
  conversation.py    rewrite follow-ups into standalone queries
api/               FastAPI + SSE
web/               React + TypeScript SPA (Vite)
eval/              golden set, harnesses, judge, RESULTS.md
index_repo.py      build the indexes
query.py / ask.py  CLI
```

## Known limitations

- `paraphrase` questions (user vocabulary, no identifiers) are the weakest segment.
  Guaranteeing code half the retrieval slots displaces the docs that answer them —
  a deliberate trade, documented in RESULTS.md finding B14.
- Answer *correctness* is unmeasured. Faithfulness checks that the sources support
  each claim, not that the claim is right.
- Structural chunking covers Python, TypeScript, JavaScript, Go, Rust and Java.
  Other languages fall back to text windows. The eval set is Python-only, so
  multi-language support is verified structurally (spans, symbol extraction)
  rather than by a retrieval metric.
- Re-indexing rebuilds from scratch; there is no incremental update.
- Background jobs run in-process (FastAPI `BackgroundTasks`). Fine for one user;
  a restart mid-index leaves a repo stuck in `indexing`.
- No auth. Public GitHub repos only, capped at 400MB and 6000 indexable files.
