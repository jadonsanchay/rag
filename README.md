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

Clone a repository to index. Repos added through the API/UI land under
`data/repos/` by default (override with `APP_DATA_DIR`, which also relocates
the vector store, lexical index, and registry — see **Deploy** below); for a
manual CLI clone, anywhere on disk works:

```bash
mkdir -p data/repos
git clone --depth 1 --branch 0.115.0 \
  https://github.com/fastapi/fastapi.git data/repos/fastapi
```

## Usage

**Index** — walk, chunk, embed, and build both indexes:

```bash
uv run python index_repo.py data/repos/fastapi \
  --include fastapi --include docs/en --variant astcode-cards
```

`--include` restricts the corpus to named subtrees, which keeps an eval corpus
reproducible. FastAPI ships 26 translations of its docs and 701 near-duplicate
tutorial snippets; indexing them floods the index with near-identical chunks.

Any repo works, not just the sample. The manifest records where each indexed tree
lives, so a repo outside `data/repos/` still resolves for citation verification
and the source viewer:

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

| endpoint | auth? | purpose |
|---|---|---|
| `POST /api/auth/signup` | — | create an account, sets the session cookie |
| `POST /api/auth/login` | — | sets the session cookie |
| `POST /api/auth/logout` | — | deletes the session, clears the cookie |
| `GET /api/auth/me` | required | current user, or 401 |
| `POST /api/repos` | required | add a public GitHub URL; clones and indexes in the background (202), or attaches to an already-indexed copy instantly (see **Authentication**) |
| `GET /api/repos` | required | *your* indexed repos |
| `GET /api/repos/{id}` | required | poll the indexing stage: `queued → cloning → indexing → ready` |
| `POST /api/repos/{id}/reindex` | required | re-clone and rebuild |
| `DELETE /api/repos/{id}` | required | drop your row (and the shared index too, if nobody else still references it) |
| `POST /api/conversations` | required | start a thread against one of your repos |
| `POST /api/conversations/{id}/messages` | required | SSE: `rewrite` → `trace` → `token`… → `done` |
| `POST /api/ask` | required | stateless single question; SSE `trace` → `token`… → `done` |
| `GET /api/file` | — | a slice of a source file, for the viewer behind a citation |
| `GET /health` | — | liveness (unprefixed — infra checks hit this directly) |
| `GET /metrics` | — | in-process counters: requests, refusals, mean retrieval/generation ms |

Everything the frontend calls lives under `/api`; `/health` and `/metrics` stay
unprefixed since they're for infra/you, not the SPA. In production the same
FastAPI process also serves the built SPA at `/` (see **Deploy**), so `/api/*`,
`/health`, `/metrics`, and the app itself are all one origin, no CORS involved.

`/api/ask` emits the retrieval trace **before** generation starts. Measured warm:
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

`/` is a landing page explaining what the tool does; "Try it now" routes to `/app`
(`react-router-dom`, one of two new dependencies in this session's work — see
**Authentication**), which requires an account and redirects to `/login`
otherwise. Once in, paste a GitHub URL to index a repo, switch between your
own indexed repos, then chat. Each
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
npm run smoke          # 12 browser checks
```

The smoke test asserts the thing unit tests cannot: that sources are on screen
*while the answer element does not yet exist*, that citation spans verify, that
clicking a marker highlights its source, and that an unanswerable question is
refused rather than answered.

## Authentication

Real accounts, not a cosmetic login screen: signup/login/logout, session-based
(`pipeline/registry.py`'s `sessions` table — a random token in an `HttpOnly`,
`SameSite=Lax` cookie, not a JWT). Passwords are hashed with `bcrypt` (the
other new dependency this session added). Logout deletes the session row —
real, immediate revocation, which a stateless JWT can't do without a
blocklist. No CSRF token: the cookie is `HttpOnly` + `SameSite=Lax` and every
mutating request needs a JSON body, which a cross-site HTML `<form>` can't
send — a standard, sufficient mitigation at this scale.

**Repos are per-user visible, but the index is shared.** If you index a repo
someone else already has `ready`, adding the same URL+variant gives you your
own row over their already-built index instantly — no re-cloning, no
re-embedding, no doubled OpenAI spend for identical content. Conversations
stay strictly yours: a conversation id that exists but isn't yours returns
`404`, not `403` — same non-disclosure GitHub uses for private repos.
Deleting your repo only tears down the underlying Chroma collection, lexical
index, and clone if no one else's row still references it.

One accepted gap, stated plainly: if someone else's index for the same
collection is still *mid-index* (not yet `ready`) when you add it, this
doesn't link your request to their in-flight job — it starts its own. Two
people independently indexing the same brand-new repo within the same few
minutes is rare and self-correcting (later additions reuse whichever copy
finishes first); the cross-job linking to close that gap isn't worth the
complexity here.

`/api/auth/login` and `/api/auth/signup` share the same per-IP rate limiter as
the cost-bearing endpoints (`api/ratelimit.py`) — a different threat model
(credential stuffing / mass account creation) reusing the same mechanism.

## Observability

Structured JSON logs to stdout (stdlib `logging`, no new dependency —
`pipeline/logging_config.py`), plus an in-process `GET /metrics`. Deliberately
not a Prometheus/Grafana stack: nothing scrapes it, and standing one up would be
theater for a single-instance project. If this runs somewhere with a real log
viewer (Fly and Railway both show stdout), JSON lines are immediately useful
there for free.

What's logged, one line per event: every HTTP request (method, path, status,
duration), every `/api/ask` and `/api/conversations/{id}/messages` answer
(refused?, retrieval ms, generation ms, invalid-citation count), every ingest
job's stage transitions and failures, and a full traceback for anything that
raises inside a request.

```bash
curl localhost:8000/metrics
# {"requests_total": 42, "requests_by_status": {"200": 40, "429": 2}, ...
#  "answers_total": 12, "refusals_total": 1,
#  "mean_retrieval_ms": 1840.2, "mean_generation_ms": 2310.5,
#  "citation_invalid_total": 0}
```

Counters are in-memory and reset on restart — enough to answer "is this healthy
right now," not a historical record.

## Deploy

One container serves both the API and the built frontend (`api/main.py`), so
there's one image, one deploy, one URL, and no CORS to configure in production.
The SPA has a real client-side route (`/` landing, `/app` the tool), so the
fallback route serves real files as themselves and `index.html` for everything
else — a direct load or refresh at `/app` still resolves; the router takes it
from there.

```bash
npm --prefix web run build        # -> web/dist, baked into the image below
docker build -t codebase-qa .
docker run -p 8000:8000 --env-file .env -v codebase_qa_data:/data codebase-qa
```

Deploying to [Fly.io](https://fly.io) (the `fly.toml` here targets it; any host
with persistent disk works, since Chroma, SQLite, and cloned repos are all files):

```bash
flyctl apps create codebase-qa            # or edit fly.toml's app name first
flyctl volumes create codebase_qa_data --size 3 --region iad
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl deploy
```

**Before sharing the URL with anyone**, set a hard monthly spend cap on the
OpenAI account in the [usage limits dashboard](https://platform.openai.com/settings/organization/limits) —
five minutes, and the last line of defense if the application-level guards below
have a bug. Everything else here reduces risk; that one bounds it.

### Cost and abuse controls

- **Per-IP rate limit** on `/api/ask`, `/api/conversations/{id}/messages`, and
  `POST /api/repos` — the three endpoints that spend money (embeddings,
  generation, or a full clone+index). Hand-rolled in-memory sliding window
  (`api/ratelimit.py`, no new dependency), returns `429` with a plain message.
  `RATE_LIMIT_PER_IP` (default 20) requests per `RATE_LIMIT_WINDOW_SECONDS`
  (default 60).
- **Global request ceiling** (`GLOBAL_REQUEST_CEILING`, default 2000) on the
  same endpoints — one counter, resets on restart. Acceptable for a demo: a
  restart is rare, not an exploitable reset-the-clock loophole in practice here.
- **Ingest guards**, already enforced before any embedding spend happens:
  `MAX_REPO_MB` (400 locally, 50 in `fly.toml`) and `MAX_INDEXED_FILES` (6000
  locally, 2000 in `fly.toml`) — both env-overridable in
  `pipeline/ingest_job.py`.
- All four are environment variables, so the deployed instance can run tighter
  than local dev without a code change — see the `[env]` block in `fly.toml`.

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
  registry.py        SQLite: users, sessions, per-user repos, conversations
  ingest_job.py      clone + background indexing with guards
  conversation.py    rewrite follow-ups into standalone queries
  logging_config.py  stdlib JSON logging to stdout
api/               FastAPI + SSE
  auth.py            signup/login/logout/me, session cookies, bcrypt
  metrics.py         in-process counters behind GET /metrics
  ratelimit.py       per-IP + global in-memory rate limiting
web/               React + TypeScript SPA (Vite)
  Login.tsx / Signup.tsx / RequireAuth.tsx, AuthContext
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
- Accounts exist (see **Authentication**) but there's no email verification
  or password reset — signup is instant and self-service, appropriate for a
  demo, not for anything holding real user data.
- Public GitHub repos only, capped at `MAX_REPO_MB`/`MAX_INDEXED_FILES`
  (400MB/6000 files locally, tighter in `fly.toml`). Rate limiting (see
  **Deploy**) bounds request volume per IP, not per account.
- Reindexing only your own row still rebuilds the *shared* index everyone
  else's row points at — there's no per-user isolation of the underlying
  data, only of visibility and conversations (see **Authentication**).
