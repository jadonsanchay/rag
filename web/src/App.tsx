import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  addRepo,
  createConversation,
  deleteRepo,
  listRepoStatuses,
  reindexRepo,
  sendMessage,
  type ChatDone,
  type RepoStatus,
  type Source,
} from "./api";
import { AnswerText } from "./components/AnswerText";
import { RepoBar } from "./components/RepoBar";
import { SourceList } from "./components/SourceList";
import { SourceViewer } from "./components/SourceViewer";

const REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT";
const POLL_MS = 2000;
const IN_PROGRESS = new Set(["queued", "cloning", "indexing"]);
const EXAMPLE_QUESTIONS = [
  "What does this project do?",
  "How is the code organized?",
  "Where does the main entry point live?",
  "How is error handling done here?",
];

interface Turn {
  id: string;
  question: string;
  rewritten: string | null;
  answer: string;
  sources: Source[];
  retrievalMs: number | null;
  done: ChatDone | null;
  error: string | null;
  streaming: boolean;
}

export default function App() {
  const navigate = useNavigate();
  const [repos, setRepos] = useState<RepoStatus[]>([]);
  const [activeRepoId, setActiveRepoId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [viewing, setViewing] = useState<{ source: Source; repo: string } | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const activeRepo = repos.find((r) => r.id === activeRepoId) ?? null;
  const streaming = turns.some((t) => t.streaming);

  const refreshRepos = useCallback(async () => {
    try {
      const rows = await listRepoStatuses();
      setRepos(rows);
      setActiveRepoId((current) => {
        if (current && rows.some((r) => r.id === current)) return current;
        return rows.find((r) => r.ready)?.id ?? rows[0]?.id ?? null;
      });
    } catch {
      setNotice("Cannot reach the API. Is uvicorn running on port 8000?");
    }
  }, []);

  useEffect(() => {
    void refreshRepos();
  }, [refreshRepos]);

  // Poll only while something is actually indexing, so an idle page is quiet.
  useEffect(() => {
    if (!repos.some((r) => IN_PROGRESS.has(r.status))) return;
    const timer = setInterval(() => void refreshRepos(), POLL_MS);
    return () => clearInterval(timer);
  }, [repos, refreshRepos]);

  // A conversation is scoped to one repo; switching repos starts a new thread.
  useEffect(() => {
    if (!activeRepo?.ready) {
      setConversationId(null);
      return;
    }
    let cancelled = false;
    setTurns([]);
    createConversation(activeRepo.id)
      .then((id) => !cancelled && setConversationId(id))
      .catch(() => !cancelled && setNotice("Could not start a conversation."));
    return () => {
      cancelled = true;
    };
  }, [activeRepo?.id, activeRepo?.ready]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  const patchTurn = (id: string, patch: Partial<Turn>) =>
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));

  const ask = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !conversationId) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const turnId = `${Date.now()}`;
      setTurns((prev) => [
        ...prev,
        {
          id: turnId,
          question: trimmed,
          rewritten: null,
          answer: "",
          sources: [],
          retrievalMs: null,
          done: null,
          error: null,
          streaming: true,
        },
      ]);
      setQuestion("");

      void sendMessage(
        conversationId,
        trimmed,
        {
          onRewrite: (_original, query) => patchTurn(turnId, { rewritten: query }),
          onTrace: (sources, ms) => patchTurn(turnId, { sources, retrievalMs: ms }),
          onToken: (chunk) =>
            setTurns((prev) =>
              prev.map((t) => (t.id === turnId ? { ...t, answer: t.answer + chunk } : t)),
            ),
          onDone: (payload) =>
            patchTurn(turnId, { done: payload, answer: payload.answer, streaming: false }),
          onError: (message) => patchTurn(turnId, { error: message, streaming: false }),
        },
        controller.signal,
      );
    },
    [conversationId],
  );

  const handleAdd = async (url: string) => {
    setBusy(true);
    setNotice(null);
    try {
      const repo = await addRepo(url);
      setActiveRepoId(repo.id);
      await refreshRepos();
    } catch (error) {
      setNotice((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (id: string) => {
    await deleteRepo(id).catch((e: Error) => setNotice(e.message));
    setConversationId(null);
    await refreshRepos();
  };

  const handleReindex = async (id: string) => {
    await reindexRepo(id).catch((e: Error) => setNotice(e.message));
    await refreshRepos();
  };

  return (
    <div className="app">
      <div className="masthead">
        <button className="masthead-back" onClick={() => navigate("/")} aria-label="Back to landing page">
          ←
        </button>
        <h1>Codebase Q&amp;A</h1>
        <span className="tagline-inline">
          Ask questions about a codebase in plain English — every answer links back to the
          exact file and line it came from.
        </span>
      </div>

      <RepoBar
        repos={repos}
        activeId={activeRepoId}
        busy={busy}
        onSelect={setActiveRepoId}
        onAdd={handleAdd}
        onDelete={handleDelete}
        onReindex={handleReindex}
      />

      {notice && <p className="error">{notice}</p>}

      {!repos.length && (
        <p className="placeholder">
          Paste a public GitHub repository URL above — it will be cloned and indexed
          in the background.
        </p>
      )}

      {activeRepo && !activeRepo.ready && (
        <p className="placeholder">
          {activeRepo.status === "failed"
            ? "This repository failed to index. Fix the URL or try another."
            : "Indexing… you can ask questions as soon as it is ready."}
        </p>
      )}

      {activeRepo?.ready && !turns.length && (
        <div className="examples">
          {EXAMPLE_QUESTIONS.map((q) => (
            <button key={q} type="button" onClick={() => ask(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="transcript">
        {turns.map((turn) => {
          const invalid = new Set(
            (turn.done?.citations ?? []).filter((c) => !c.valid).map((c) => c.index),
          );
          const refused = turn.done?.refused ?? turn.answer.startsWith(REFUSAL_TOKEN);

          return (
            <div className="turn" key={turn.id}>
              <div className="bubble user">{turn.question}</div>

              {turn.rewritten && (
                /* Surfacing the rewrite matters: otherwise a user cannot tell why
                   a follow-up returned what it did. */
                <div className="rewrite-note">
                  searched for: <span>{turn.rewritten}</span>
                </div>
              )}

              <div className="turn-body">
                <div className="bubble assistant">
                  {turn.error && <p className="error">{turn.error}</p>}

                  {!turn.error && refused && (
                    <div className="refusal">
                      <strong>Not answerable from this repository.</strong>
                    </div>
                  )}

                  {!turn.error && !refused && turn.answer && (
                    <div className="answer">
                      <AnswerText
                        text={turn.answer}
                        invalid={invalid}
                        onCitationClick={(index) =>
                          document
                            .getElementById(`src-${turn.id}-${index}`)
                            ?.scrollIntoView({ behavior: "smooth", block: "center" })
                        }
                      />
                      {turn.streaming && <span className="caret" />}
                    </div>
                  )}

                  {!turn.error && !turn.answer && turn.streaming && (
                    <p className="placeholder">
                      {turn.sources.length ? "Sources retrieved — generating…" : "Retrieving…"}
                    </p>
                  )}

                  {turn.done && !refused && (
                    <div className="verify">
                      <span className={invalid.size ? "bad" : "ok"}>
                        {turn.done.citation_summary || "no citations"}
                      </span>
                      <span>
                        {turn.done.timing_ms.retrieval}ms + {turn.done.timing_ms.generation}ms
                      </span>
                    </div>
                  )}
                </div>

                {turn.sources.length > 0 && (
                  <div className="turn-sources" id={`sources-${turn.id}`}>
                    <SourceList
                      sources={turn.sources}
                      citedIndices={turn.done?.cited_indices ?? []}
                      citations={turn.done?.citations ?? []}
                      activeIndex={null}
                      idPrefix={`src-${turn.id}`}
                      onSelect={(source) =>
                        activeRepo && setViewing({ source, repo: activeRepo.name })
                      }
                    />
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      <form
        className="ask sticky"
        onSubmit={(event) => {
          event.preventDefault();
          ask(question);
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={
            turns.length
              ? "Follow up — pronouns are resolved against the conversation…"
              : "Ask about this repository…"
          }
          aria-label="Question"
          disabled={!conversationId}
        />
        <button type="submit" disabled={streaming || !question.trim() || !conversationId}>
          {streaming ? "…" : "Send"}
        </button>
        {streaming && (
          <button
            type="button"
            className="secondary"
            onClick={() => {
              abortRef.current?.abort();
              setTurns((prev) => prev.map((t) => ({ ...t, streaming: false })));
            }}
          >
            Stop
          </button>
        )}
      </form>

      {viewing && (
        <SourceViewer
          source={viewing.source}
          repo={viewing.repo}
          onClose={() => setViewing(null)}
        />
      )}
    </div>
  );
}
