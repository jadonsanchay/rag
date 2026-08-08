import { useEffect, useRef, useState } from "react";
import { fetchFile, type FileSlice, type Source } from "../api";

interface Props {
  source: Source;
  repo: string;
  onClose(): void;
}

/**
 * Slide-over showing the cited file, with the cited lines highlighted.
 *
 * This closes the verification loop: a citation is only trustworthy if you can see
 * the code behind it without leaving the page.
 */
export function SourceViewer({ source, repo, onClose }: Props) {
  const [slice, setSlice] = useState<FileSlice | null>(null);
  const [error, setError] = useState<string | null>(null);
  const highlightRef = useRef<HTMLDivElement | null>(null);

  const isDirectory = source.path.endsWith("/");
  const from = source.start_line ?? 1;
  const to = source.end_line ?? from;

  useEffect(() => {
    if (isDirectory) return;
    let cancelled = false;
    setSlice(null);
    setError(null);

    fetchFile(source.path, from, to, repo)
      .then((data) => !cancelled && setSlice(data))
      .catch((err: Error) => !cancelled && setError(err.message));

    return () => {
      cancelled = true;
    };
  }, [source.path, from, to, repo, isDirectory]);

  // Escape to dismiss — expected of any overlay.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    highlightRef.current?.scrollIntoView({ block: "center" });
  }, [slice]);

  return (
    <div className="viewer-backdrop" onClick={onClose}>
      <div className="viewer" onClick={(event) => event.stopPropagation()}>
        <div className="viewer-head">
          <span className="path">
            {source.path}
            {!isDirectory && source.start_line ? `:${from}-${to}` : ""}
          </span>
          <button className="close" onClick={onClose}>
            Esc
          </button>
        </div>

        <div className="viewer-body">
          {isDirectory && (
            <p className="placeholder" style={{ padding: "0 16px" }}>
              This is a package card — a generated summary of the{" "}
              <code>{source.path}</code> directory, not a file on disk.
              <br />
              <br />
              {source.preview}
            </p>
          )}
          {error && <p className="error">{error}</p>}
          {!isDirectory && !slice && !error && (
            <p className="placeholder" style={{ padding: "0 16px" }}>
              Loading…
            </p>
          )}

          {slice &&
            slice.content.split("\n").map((line, offset) => {
              const lineNumber = slice.start_line + offset;
              const highlighted = lineNumber >= from && lineNumber <= to;
              return (
                <div
                  key={lineNumber}
                  className={highlighted ? "code-line hl" : "code-line"}
                  ref={highlighted && lineNumber === from ? highlightRef : undefined}
                >
                  <span className="ln">{lineNumber}</span>
                  <span className="src">{line || " "}</span>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}
