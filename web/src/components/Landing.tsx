interface Props {
  onStart(): void;
}

const FEATURES = [
  {
    title: "Verified citations",
    body: "Every [n] in an answer links to a real file:line span, checked against the working tree before it's shown as verified.",
  },
  {
    title: "Hybrid retrieval",
    body: "Keyword search and meaning-based search run together, so exact identifiers and natural-language questions both work.",
  },
  {
    title: "Honest refusals",
    body: "If the repository doesn't contain the answer, it says so instead of guessing.",
  },
];

export function Landing({ onStart }: Props) {
  return (
    <div className="landing-page">
      <div className="landing-hero">
        <h1>Codebase Q&amp;A</h1>
        <p className="landing-sub">
          Point it at a public GitHub repository and ask questions about it in plain
          English. Every answer comes back with citations to the exact file and line
          it came from.
        </p>
        <button className="landing-cta" onClick={onStart}>
          Try it now
        </button>
      </div>

      <ol className="how-it-works landing-steps">
        <li>
          <strong>Retrieve</strong> — relevant code and docs are pulled from the repo
          using both keyword and meaning-based search.
        </li>
        <li>
          <strong>Generate</strong> — an answer is written from only that retrieved
          material, with inline citations.
        </li>
        <li>
          <strong>Verify</strong> — every citation is checked against the real file
          before it's shown as verified.
        </li>
      </ol>

      <div className="landing-features">
        {FEATURES.map((feature) => (
          <div className="landing-feature" key={feature.title}>
            <h3>{feature.title}</h3>
            <p>{feature.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
