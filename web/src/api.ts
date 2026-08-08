const API = "/api";

export interface Source {
  index: number;
  path: string;
  start_line: number | null;
  end_line: number | null;
  symbol: string | null;
  kind: string | null;
  language: string | null;
  score: number;
  /** Rank within each retrieval list, e.g. { "lexical:code": 1 }. */
  ranks: Record<string, number>;
  retrievers: string[];
  preview: string;
}

export interface Citation {
  index: number;
  path: string | null;
  start_line: number | null;
  end_line: number | null;
  valid: boolean;
  problem: string | null;
}

export interface AnswerDone {
  answer: string;
  refused: boolean;
  cited_indices: number[];
  citations: Citation[];
  fabricated_indices: number[];
  citation_summary: string;
  timing_ms: { retrieval?: number; generation?: number };
}

export interface FileSlice {
  path: string;
  start_line: number;
  end_line: number;
  total_lines: number;
  language: string | null;
  content: string;
}

export interface Repo {
  repo: string;
  variant: string;
  collection: string;
  chunks: number;
  files_indexed: number | null;
  commit_sha: string | null;
  embedding_model: string | null;
}

export interface AskHandlers {
  onTrace(sources: Source[], retrievalMs: number): void;
  onToken(text: string): void;
  onDone(done: AnswerDone): void;
  onError(message: string): void;
}

export interface AskOptions {
  question: string;
  repo?: string;
  variant?: string;
  mode?: "semantic" | "lexical" | "hybrid";
  top_k?: number;
  signal?: AbortSignal;
}

/**
 * Stream an answer.
 *
 * EventSource cannot issue a POST, and the question does not belong in a query
 * string, so SSE frames are parsed by hand off the fetch body stream. Frames are
 * separated by a blank line and can straddle chunk boundaries, hence the buffer.
 */
export async function askStream(
  options: AskOptions,
  handlers: AskHandlers,
): Promise<void> {
  const { signal, ...body } = options;

  let response: Response;
  try {
    response = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    handlers.onError(`Could not reach the API: ${(error as Error).message}`);
    return;
  }

  if (!response.ok) {
    const detail = await response.text();
    handlers.onError(`HTTP ${response.status}: ${detail.slice(0, 400)}`);
    return;
  }
  if (!response.body) {
    handlers.onError("Response had no body to stream.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string) => {
    let event = "message";
    const dataLines: string[] = [];

    for (const line of frame.split("\n")) {
      if (line.startsWith(":")) continue; // comment / keep-alive
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;

    let payload: unknown;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }

    switch (event) {
      case "trace": {
        const data = payload as { sources: Source[]; retrieval_ms: number };
        handlers.onTrace(data.sources, data.retrieval_ms);
        break;
      }
      case "token":
        handlers.onToken((payload as { text: string }).text);
        break;
      case "done":
        handlers.onDone(payload as AnswerDone);
        break;
      case "error": {
        const data = payload as { message: string; detail?: string };
        handlers.onError(data.detail ? `${data.message}: ${data.detail}` : data.message);
        break;
      }
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        dispatch(buffer.slice(0, split));
        buffer = buffer.slice(split + 2);
        split = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (error) {
    if ((error as Error).name !== "AbortError") {
      handlers.onError(`Stream failed: ${(error as Error).message}`);
    }
  }
}

export async function fetchFile(
  path: string,
  start: number,
  end: number,
  repo = "fastapi",
  context = 25,
  variant?: string,
): Promise<FileSlice> {
  const params = new URLSearchParams({
    repo,
    ...(variant ? { variant } : {}),
    path,
    start: String(start),
    end: String(end),
    context: String(context),
  });
  const response = await fetch(`${API}/file?${params}`);
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status})`);
  return response.json();
}

export async function fetchRepos(): Promise<Repo[]> {
  const response = await fetch(`${API}/repos`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// --- steps 10-12: repo lifecycle and conversations -------------------------

export interface RepoStatus {
  id: string;
  name: string;
  url: string | null;
  variant: string;
  collection: string;
  status: "queued" | "cloning" | "indexing" | "ready" | "failed";
  stage: string | null;
  error: string | null;
  commit_sha: string | null;
  files_indexed: number;
  chunks: number;
  languages: Record<string, number>;
  ready: boolean;
}

export interface ChatDone extends AnswerDone {
  message_id?: string;
}

export interface ChatHandlers extends AskHandlers {
  /** Fires only when a follow-up was condensed into a standalone query. */
  onRewrite(original: string, query: string): void;
}

export async function listRepoStatuses(): Promise<RepoStatus[]> {
  const response = await fetch(`${API}/repos`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

export async function addRepo(url: string, variant = "main"): Promise<RepoStatus> {
  const response = await fetch(`${API}/repos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, variant }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`,
    );
  }
  return body;
}

export async function getRepo(id: string): Promise<RepoStatus> {
  const response = await fetch(`${API}/repos/${id}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

export async function deleteRepo(id: string): Promise<void> {
  const response = await fetch(`${API}/repos/${id}`, { method: "DELETE" });
  if (!response.ok && response.status !== 204) {
    throw new Error(`HTTP ${response.status}`);
  }
}

export async function reindexRepo(id: string): Promise<RepoStatus> {
  const response = await fetch(`${API}/repos/${id}/reindex`, { method: "POST" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

export async function createConversation(repoId: string): Promise<string> {
  const response = await fetch(`${API}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id: repoId }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()).id;
}

/** Send a chat turn. Same SSE parsing as askStream, plus the `rewrite` frame. */
export async function sendMessage(
  conversationId: string,
  question: string,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API}/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 6 }),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    handlers.onError(`Could not reach the API: ${(error as Error).message}`);
    return;
  }

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    handlers.onError(`HTTP ${response.status}: ${detail.slice(0, 300)}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;

    let payload: unknown;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }

    switch (event) {
      case "rewrite": {
        const data = payload as { original: string; query: string };
        handlers.onRewrite(data.original, data.query);
        break;
      }
      case "trace": {
        const data = payload as { sources: Source[]; retrieval_ms: number };
        handlers.onTrace(data.sources, data.retrieval_ms);
        break;
      }
      case "token":
        handlers.onToken((payload as { text: string }).text);
        break;
      case "done":
        handlers.onDone(payload as ChatDone);
        break;
      case "error": {
        const data = payload as { message: string; detail?: string };
        handlers.onError(data.detail ? `${data.message}: ${data.detail}` : data.message);
        break;
      }
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        dispatch(buffer.slice(0, split));
        buffer = buffer.slice(split + 2);
        split = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (error) {
    if ((error as Error).name !== "AbortError") {
      handlers.onError(`Stream failed: ${(error as Error).message}`);
    }
  }
}
