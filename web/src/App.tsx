import { useCallback, useEffect, useRef, useState } from "react";
import {
  askStream,
  fetchRepos,
  type AnswerDone,
  type Repo,
  type Source,
} from "./api";
import { AnswerText } from "./components/AnswerText";
import { SourceList } from "./components/SourceList";
import { SourceViewer } from "./components/SourceViewer";

const REPO = "fastapi";
const VARIANT = "astcode-cards";

const EXAMPLES = [
  "Where is the APIRouter class defined?",
  "How is the security system organized?",
  "How does FastAPI decide to run my endpoint in a threadpool?",
  "How do I let someone upload a file?",
  "How does the billing module calculate charges?",
];

const REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT";

export default function App() {
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [answer, setAnswer] = useState("");
  const [done, setDone] = useState<AnswerDone | null>(null);
  const [retrievalMs, setRetrievalMs] = useState<number | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewing, setViewing] = useState<Source | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [repo, setRepo] = useState<Repo | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchRepos()
      .then((repos) => setRepo(repos.find((r) => r.variant === VARIANT) ?? null))
      .catch(() => setRepo(null));
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const ask = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setQuestion(trimmed);
    setSources([]);
    setAnswer("");
    setDone(null);
    setRetrievalMs(null);
    setError(null);
    setActiveIndex(null);
    setStreaming(true);

    void askStream(
      { question: trimmed, repo: REPO, variant: VARIANT, signal: controller.signal },
      {
        onTrace: (incoming, ms) => {
          // Sources land before generation starts, so they render immediately
          // rather than after the answer completes.
          setSources(incoming);
          setRetrievalMs(ms);
        },
        onToken: (chunk) => setAnswer((prev) => prev + chunk),
        onDone: (payload) => {
          setDone(payload);
          setAnswer(payload.answer);
          setStreaming(false);
        },
        onError: (message) => {
          setError(message);
          setStreaming(false);
        },
      },
    );
  }, []);

  const jumpToSource = useCallback((index: number) => {
    setActiveIndex(index);
    document
      .getElementById(`source-${index}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const invalidCitations = new Set(
    (done?.citations ?? []).filter((c) => !c.valid).map((c) => c.index),
  );
  const refused = done?.refused ?? answer.startsWith(REFUSAL_TOKEN);
  const refusalDetail = refused
    ? answer.slice(answer.indexOf(REFUSAL_TOKEN) + REFUSAL_TOKEN.length).trim()
    : "";

  return (
    <div className="app">
      <div className="masthead">
        <h1>Codebase Q&amp;A</h1>
        {repo && (
          <span className="repo-chip">
            {repo.repo} @ {repo.commit_sha?.slice(0, 7)} · {repo.chunks} chunks
          </span>
        )}
      </div>
      <p className="tagline">
        Hybrid retrieval over source and docs. Every citation is checked against the
        working tree, and the trace shows why each source ranked.
      </p>

      <form
        className="ask"
        onSubmit={(event) => {
          event.preventDefault();
          ask(question);
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about the indexed repository…"
          aria-label="Question"
        />
        <button type="submit" disabled={streaming || !question.trim()}>
          {streaming ? "Answering…" : "Ask"}
        </button>
        {streaming && (
          <button
            type="button"
            className="secondary"
            onClick={() => {
              abortRef.current?.abort();
              setStreaming(false);
            }}
          >
            Stop
          </button>
        )}
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button key={example} disabled={streaming} onClick={() => ask(example)}>
            {example}
          </button>
        ))}
      </div>

      <div className="columns">
        <section className="panel">
          <div className="panel-head">
            <h2>Answer</h2>
            {done && (
              <span style={{ font: "11px var(--mono)", color: "var(--muted)" }}>
                {done.timing_ms.retrieval}ms retrieval · {done.timing_ms.generation}ms
                generation
              </span>
            )}
          </div>
          <div className="panel-body">
            {error && <p className="error">{error}</p>}

            {!error && !answer && !streaming && (
              <p className="placeholder">
                Ask a question, or try one of the examples. The last example is
                deliberately unanswerable — the system should refuse rather than
                invent something.
              </p>
            )}

            {!error && refused && (
              <div className="refusal">
                <strong>Not answerable from this repository.</strong>
                {refusalDetail && <div style={{ marginTop: 4 }}>{refusalDetail}</div>}
              </div>
            )}

            {!error && !refused && answer && (
              <div className="answer">
                <AnswerText
                  text={answer}
                  invalid={invalidCitations}
                  onCitationClick={jumpToSource}
                />
                {streaming && <span className="caret" />}
              </div>
            )}

            {!error && !answer && streaming && (
              <p className="placeholder">
                {sources.length
                  ? "Sources retrieved — generating…"
                  : "Retrieving…"}
              </p>
            )}

            {done && !refused && (
              <div className="verify" style={{ marginTop: 16 }}>
                <span className={invalidCitations.size ? "bad" : "ok"}>
                  {done.citation_summary || "no citations"}
                </span>
                {done.fabricated_indices.length > 0 && (
                  <span className="bad">
                    fabricated source numbers: {done.fabricated_indices.join(", ")}
                  </span>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Retrieved sources</h2>
            {retrievalMs !== null && (
              <span style={{ font: "11px var(--mono)", color: "var(--muted)" }}>
                {sources.length} in {retrievalMs}ms
              </span>
            )}
          </div>
          {sources.length ? (
            <SourceList
              sources={sources}
              citedIndices={done?.cited_indices ?? []}
              citations={done?.citations ?? []}
              activeIndex={activeIndex}
              onSelect={setViewing}
            />
          ) : (
            <div className="panel-body">
              <p className="placeholder">
                Sources appear here as soon as retrieval finishes — before the answer
                is written. Click one to read the code.
              </p>
            </div>
          )}
        </section>
      </div>

      {viewing && (
        <SourceViewer source={viewing} repo={REPO} onClose={() => setViewing(null)} />
      )}
    </div>
  );
}
