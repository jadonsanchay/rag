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
): Promise<FileSlice> {
  const params = new URLSearchParams({
    repo,
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
