import { Fragment, type ReactNode } from "react";

const CITATION = /\[(\d{1,2})\]/g;
const INLINE_CODE = /`([^`\n]+)`/g;

interface Props {
  text: string;
  /** Citation indices that failed span verification, rendered as invalid. */
  invalid: Set<number>;
  onCitationClick(index: number): void;
}

/** Render backtick spans as <code>, leaving other text untouched. */
function withInlineCode(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;

  INLINE_CODE.lastIndex = 0;
  while ((match = INLINE_CODE.exec(text)) !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index));
    nodes.push(<code key={`${keyPrefix}-c${match.index}`}>{match[1]}</code>);
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

/**
 * The answer, with [n] markers turned into buttons that jump to the source.
 *
 * Rendering runs on every token during streaming, so it stays a plain regex walk
 * rather than a markdown parser — a half-streamed answer is frequently invalid
 * markdown (unclosed fences, dangling backticks), and a parser would flicker.
 */
export function AnswerText({ text, invalid, onCitationClick }: Props) {
  const parts: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;

  CITATION.lastIndex = 0;
  while ((match = CITATION.exec(text)) !== null) {
    if (match.index > cursor) {
      parts.push(
        <Fragment key={`t${cursor}`}>
          {withInlineCode(text.slice(cursor, match.index), `t${cursor}`)}
        </Fragment>,
      );
    }

    const index = Number(match[1]);
    const bad = invalid.has(index);
    parts.push(
      <button
        key={`cite-${match.index}`}
        className={bad ? "cite invalid" : "cite"}
        onClick={() => onCitationClick(index)}
        title={bad ? `Source ${index} failed verification` : `Jump to source ${index}`}
      >
        {index}
      </button>,
    );
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) {
    parts.push(
      <Fragment key={`t${cursor}`}>
        {withInlineCode(text.slice(cursor), `t${cursor}`)}
      </Fragment>,
    );
  }

  return <>{parts}</>;
}
