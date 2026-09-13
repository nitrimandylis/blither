import { useEffect, useState, type CSSProperties } from "react";
import { checkName, type Check } from "./lib/check";
import { generate, randomSeed, type Style } from "./lib/generate";
import { loadModel, type Model } from "./lib/model";

const CATEGORIES = ["cli", "app", "startup", "any"] as const;
const STYLES: Style[] = ["sensible", "bold", "unhinged"];
const BATCH = 5;

type Category = (typeof CATEGORIES)[number];

function readUrl() {
  const params = new URLSearchParams(location.search);
  const category = params.get("for") as Category | null;
  const style = params.get("style") as Style | null;
  const seed = Number(params.get("seed"));
  return {
    category: category && CATEGORIES.includes(category) ? category : "cli",
    style: style && STYLES.includes(style) ? style : "sensible",
    seed: Number.isInteger(seed) && seed > 0 ? seed : randomSeed(),
  };
}

// The "any" batch mixes categories, so guess which one a name came from for the checks.
function categoryFor(name: string, category: Category): string {
  if (category !== "any") return category;
  if (name.includes(" ") || name.endsWith(".ai")) return "startup";
  return "cli";
}

// Types the name out one character at a time. Skipped when the user prefers reduced motion.
function useTyped(text: string): string {
  const [shown, setShown] = useState(text);
  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(text);
      return;
    }
    let i = 0;
    setShown("");
    const timer = setInterval(() => {
      i += 1;
      setShown(text.slice(0, i));
      if (i >= text.length) clearInterval(timer);
    }, 38);
    return () => clearInterval(timer);
  }, [text]);
  return shown;
}

export default function App() {
  const [model, setModel] = useState<Model | null>(null);
  const [{ category, style, seed }, setState] = useState(readUrl);
  const [names, setNames] = useState<string[]>([]);
  const [pick, setPick] = useState(0);
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadModel().then(setModel);
  }, []);

  // Regenerate whenever the inputs change, and keep the URL shareable.
  useEffect(() => {
    if (!model) return;
    setNames(generate(model, category, style, BATCH, seed));
    setPick(0);
    history.replaceState(null, "", `?for=${category}&style=${style}&seed=${seed}`);
  }, [model, category, style, seed]);

  const name = names[pick] ?? "";
  const typed = useTyped(name);

  useEffect(() => {
    if (!name) return;
    setChecks(null);
    setCopied(false);
    let cancelled = false;
    checkName(name, categoryFor(name, category)).then((result) => {
      if (!cancelled) setChecks(result);
    });
    return () => {
      cancelled = true;
    };
  }, [name, category]);

  function copy() {
    navigator.clipboard.writeText(name).then(() => setCopied(true));
  }

  const subject = category === "any" ? "thing" : category;

  return (
    <main>
      <header>
        <span className="wordmark">blither</span>
        <a href="https://github.com/nitrimandylis/blither">source</a>
      </header>

      <section className="hero" aria-live="polite">
        <p className="lead">your next {subject} is called</p>
        {model ? (
          <h1 className="name" style={{ "--len": Math.max(name.length, 6) } as CSSProperties}>
            {typed}
            <span className="caret" aria-hidden="true" />
          </h1>
        ) : (
          <h1 className="name muted" style={{ "--len": 17 } as CSSProperties}>loading the model</h1>
        )}
        <ul className="checks">
          {(checks ?? []).map((c) => (
            <li key={c.label}>
              <a href={c.url} target="_blank" rel="noreferrer">{c.label}</a>
              <span className={`status ${c.status}`}>{c.status}</span>
            </li>
          ))}
          {name && !checks && <li className="muted">checking registries</li>}
        </ul>
      </section>

      <section className="controls">
        <fieldset>
          <legend>for a</legend>
          {CATEGORIES.map((c) => (
            <label key={c}>
              <input type="radio" name="category" value={c} checked={category === c}
                onChange={() => setState((s) => ({ ...s, category: c }))} />
              {c}
            </label>
          ))}
        </fieldset>
        <fieldset>
          <legend>sounding</legend>
          {STYLES.map((s) => (
            <label key={s}>
              <input type="radio" name="style" value={s} checked={style === s}
                onChange={() => setState((prev) => ({ ...prev, style: s }))} />
              {s}
            </label>
          ))}
        </fieldset>
        <div className="actions">
          <button type="button" onClick={() => setState((s) => ({ ...s, seed: randomSeed() }))} disabled={!model}>
            another
          </button>
          <button type="button" className="quiet" onClick={copy} disabled={!name}>
            {copied ? "copied" : "copy"}
          </button>
        </div>
      </section>

      {names.length > 1 && (
        <section className="batch">
          <p className="lead">also in this batch</p>
          <ul>
            {names.map((n, i) =>
              i === pick ? null : (
                <li key={n}>
                  <button type="button" className="link" onClick={() => setPick(i)}>{n}</button>
                </li>
              ),
            )}
          </ul>
        </section>
      )}

      <footer>
        <p>
          A 49k-parameter character model trained on {model?.manifest.training.names.toLocaleString() ?? "18,947"} names
          from Homebrew and the YC directory. It runs in your browser: no LLM, no server, and the seed in the URL
          reproduces this exact batch.
        </p>
        <p className="muted">blither is the model's own output (cli, sensible, seed 2, ninth of ten). Not affiliated with Homebrew or Y Combinator.</p>
      </footer>
    </main>
  );
}
