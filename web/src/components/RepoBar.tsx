import { useState } from "react";
import type { RepoStatus } from "../api";

interface Props {
  repos: RepoStatus[];
  activeId: string | null;
  busy: boolean;
  onSelect(id: string): void;
  onAdd(url: string): void;
  onDelete(id: string): void;
  onReindex(id: string): void;
}

const IN_PROGRESS = new Set(["queued", "cloning", "indexing"]);

function label(repo: RepoStatus): string {
  return repo.variant && repo.variant !== "main"
    ? `${repo.name} · ${repo.variant}`
    : repo.name;
}

export function RepoBar({
  repos,
  activeId,
  busy,
  onSelect,
  onAdd,
  onDelete,
  onReindex,
}: Props) {
  const [url, setUrl] = useState("");
  const active = repos.find((r) => r.id === activeId) ?? null;

  return (
    <div className="repo-bar">
      <div className="repo-chips">
        {repos.map((repo) => {
          const classes = ["repo-tab"];
          if (repo.id === activeId) classes.push("active");
          if (!repo.ready) classes.push("pending");
          if (repo.status === "failed") classes.push("failed");
          return (
            <button
              key={repo.id}
              className={classes.join(" ")}
              onClick={() => onSelect(repo.id)}
              title={repo.error ?? repo.collection}
            >
              {label(repo)}
              {IN_PROGRESS.has(repo.status) && <span className="spinner" />}
              {repo.status === "failed" && <span className="dot bad" />}
              {repo.ready && <span className="count">{repo.chunks}</span>}
            </button>
          );
        })}
      </div>

      <form
        className="repo-add"
        onSubmit={(event) => {
          event.preventDefault();
          if (url.trim()) {
            onAdd(url.trim());
            setUrl("");
          }
        }}
      >
        <input
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          placeholder="https://github.com/owner/repo"
          aria-label="GitHub repository URL"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !url.trim()}>
          Index
        </button>
      </form>

      {active && (
        <div className="repo-detail">
          {/* Progress is a stage name rather than a percentage: the stages have
              wildly different durations, so a bar would lie. */}
          {IN_PROGRESS.has(active.status) && (
            <span className="stage">
              {active.status}
              {active.stage ? ` — ${active.stage}` : ""}
            </span>
          )}
          {active.status === "failed" && (
            <span className="stage bad">failed — {active.error}</span>
          )}
          {active.ready && (
            <span className="stage">
              {active.files_indexed} files · {active.chunks} chunks
              {active.commit_sha ? ` · ${active.commit_sha.slice(0, 7)}` : ""}
              {Object.keys(active.languages).length
                ? ` · ${Object.entries(active.languages)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 3)
                    .map(([lang, n]) => `${lang} ${n}`)
                    .join(", ")}`
                : ""}
            </span>
          )}
          <span className="repo-actions">
            <button onClick={() => onReindex(active.id)} disabled={!active.ready}>
              Re-index
            </button>
            <button
              className="danger"
              onClick={() => {
                if (confirm(`Remove ${label(active)} and its indexes?`)) {
                  onDelete(active.id);
                }
              }}
            >
              Remove
            </button>
          </span>
        </div>
      )}
    </div>
  );
}
