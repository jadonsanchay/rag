import type { Citation, Source } from "../api";

interface Props {
  sources: Source[];
  citedIndices: number[];
  citations: Citation[];
  activeIndex: number | null;
  /** Ids must be unique per turn once several turns are on screen. */
  idPrefix?: string;
  onSelect(source: Source): void;
}

const CODE_LANGUAGES = new Set([
  "python", "javascript", "typescript", "go", "java", "ruby", "rust",
  "c", "cpp", "csharp", "php", "swift", "kotlin", "scala", "shell", "sql",
]);

function location(source: Source): string {
  if (source.path.endsWith("/")) return source.path;
  if (source.start_line && source.end_line) {
    return `${source.path}:${source.start_line}-${source.end_line}`;
  }
  return source.path;
}

/**
 * Turn the rank map into a readable explanation.
 *
 * A chunk can rank 1st among code but 15th overall, and that gap is exactly why
 * stratified retrieval finds it. This detail is real evidence, not noise — but
 * it's also implementation vocabulary a first-time visitor hasn't been given, so
 * it renders behind a per-source disclosure (see `<details>` below) rather than
 * always-on.
 */
function why(source: Source): string {
  const entries = Object.entries(source.ranks);
  if (!entries.length) return source.retrievers.join(", ") || "—";
  return entries
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([list, rank]) => `${list} #${rank}`)
    .join("  ·  ");
}

export function SourceList({
  sources,
  citedIndices,
  citations,
  activeIndex,
  idPrefix = "source",
  onSelect,
}: Props) {
  const cited = new Set(citedIndices);
  const verified = new Map(citations.map((c) => [c.index, c]));

  return (
    <div className="source-list">
      {sources.map((source) => {
        const isCited = cited.has(source.index);
        const check = verified.get(source.index);
        const isCard = source.kind === "file_card" || source.kind === "package_card";
        const isCode = source.language ? CODE_LANGUAGES.has(source.language) : false;

        const classes = ["source"];
        if (isCited) classes.push("cited");
        if (activeIndex === source.index) classes.push("active");

        return (
          <div key={source.index} id={`${idPrefix}-${source.index}`} className={classes.join(" ")}>
            <button className="source-main" onClick={() => onSelect(source)}>
              <div className="source-top">
                <span className="source-idx">{source.index}</span>
                <span className="source-path">{location(source)}</span>
                {isCard && <span className="badge card">card</span>}
                {!isCard && isCode && <span className="badge code">code</span>}
                {check?.valid && <span className="badge verified">verified</span>}
                {check && !check.valid && <span className="badge">unverified</span>}
              </div>
              <div className="source-meta">
                {source.symbol && <span>{source.symbol}</span>}
                <span>score {source.score.toFixed(4)}</span>
              </div>
            </button>

            {/* Collapsed by default: this is the retrieval-internals story from
                the README, kept, just not forced on a first-time visitor. */}
            <details className="source-details">
              <summary>Show retrieval details</summary>
              <div className="source-why">{why(source)}</div>
            </details>
          </div>
        );
      })}
    </div>
  );
}
