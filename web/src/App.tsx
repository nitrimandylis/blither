import { useEffect, useState, type CSSProperties, type FormEvent } from "react";
import { checkName, type Check } from "./lib/check";
import { cleanHints, generate, randomSeed, type Style } from "./lib/generate";
import { loadModel, type Model } from "./lib/model";
import { batchUrl, store } from "./lib/store";
import History from "./History";

const CATEGORIES = ["cli", "app", "startup"] as const;
const STYLES: Style[] = ["sensible", "bold", "unhinged"];
const BATCH = 5;

type Category = (typeof CATEGORIES)[number];

function readInitial() {
  const params = new URLSearchParams(location.search);
  const prefs = store.prefs();
  const category = (params.get("for") ?? prefs.category) as Category | null;
  const style = (params.get("style") ?? prefs.style) as Style | null;
  const seed = Number(params.get("seed"));
  return {
    category: category && CATEGORIES.includes(category) ? category : "cli",
    style: style && STYLES.includes(style) ? style : "sensible",
    seed: Number.isInteger(seed) && seed > 0 ? seed : randomSeed(),
    hints: params.get("hints") ?? "",
  };
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

// Light or dark: the remembered choice wins, otherwise the system's.
function useTheme() {
  const [theme, setTheme] = useState(() => store.prefs().theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    // the browser bar and the installed app's status bar follow the page
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#12161e" : "#eef1f4");
  }, [theme]);
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    store.setPref("theme", next);
    setTheme(next);
  }
  return { theme, toggle };
}

export default function App() {
  const { theme, toggle } = useTheme();
  const onHistory = location.pathname === "/history";
  return (
    <main>
      <header>
        <a className="wordmark" href="/">blither</a>
        <nav>
          <a href="/history" aria-current={onHistory ? "page" : undefined}>history</a>
          <a href="https://github.com/nitrimandylis/blither">source</a>
          <button type="button" className="link" onClick={toggle} aria-label={`switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            {theme === "dark" ? "light" : "dark"}
          </button>
        </nav>
      </header>
      {onHistory ? <History /> : <Namer />}
    </main>
  );
}

function Namer() {
  const [model, setModel] = useState<Model | null>(null);
  const [{ category, style, seed, hints }, setState] = useState(readInitial);
  const [draft, setDraft] = useState(hints);
  const [names, setNames] = useState<string[]>([]);
  const [pick, setPick] = useState(0);
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [copied, setCopied] = useState(false);
  const [kept, setKept] = useState(() => new Set(store.kept().map((k) => k.name)));

  useEffect(() => {
    loadModel().then(setModel);
  }, []);

  // Regenerate whenever the inputs change; keep the URL shareable and the batch in history.
  useEffect(() => {
    if (!model) return;
    const batch = generate(model, category, style, BATCH, seed, cleanHints(hints));
    setNames(batch);
    setPick(0);
    history.replaceState(null, "", batchUrl({ category, style, seed, hints }));
    store.setPref("category", category);
    store.setPref("style", style);
    store.logBatch({ names: batch, category, style, hints, seed, at: Date.now() });
  }, [model, category, style, seed, hints]);

  const name = names[pick] ?? "";
  const typed = useTyped(name);

  useEffect(() => {
    if (!name) return;
    setChecks(null);
    setCopied(false);
    let cancelled = false;
    checkName(name, category).then((result) => {
      if (!cancelled) setChecks(result);
    });
    return () => {
      cancelled = true;
    };
  }, [name, category]);

  function applyHints(event: FormEvent) {
    event.preventDefault();
    setState((s) => ({ ...s, hints: cleanHints(draft).join(" "), seed: randomSeed() }));
  }

  function copy() {
    navigator.clipboard.writeText(name).then(() => setCopied(true));
  }

  function keep() {
    if (kept.has(name)) {
      store.unkeep(name);
    } else {
      store.keep({ name, category, style, hints, seed, at: Date.now() });
    }
    setKept(new Set(store.kept().map((k) => k.name)));
  }

  return (
    <>
      <section className="hero" aria-live="polite">
        <p className="lead">your next {category} is called</p>
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
        <form className="hints" onSubmit={applyHints}>
          <label htmlFor="hints">about</label>
          <input id="hints" value={draft} onChange={(e) => setDraft(e.target.value)}
            placeholder="a few words on what it does" maxLength={60} autoComplete="off" spellCheck={false} />
          <button type="submit" className="quiet" disabled={!model || cleanHints(draft).join(" ") === hints}>name it</button>
        </form>
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
          <button type="button" className="quiet" onClick={keep} disabled={!name} aria-pressed={kept.has(name)}>
            {kept.has(name) ? "kept" : "keep"}
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
                  <button type="button" className="link mono" onClick={() => setPick(i)}>{n}</button>
                </li>
              ),
            )}
          </ul>
        </section>
      )}

      <footer>
        <p>
          A 49k-parameter character model trained on {model?.manifest.training.names.toLocaleString() ?? "14,335"} names
          from Homebrew and the YC directory. It runs in your browser: no LLM, no server, and the seed in the URL
          reproduces this exact batch.
        </p>
        <p className="muted">blither is what the first version of this model called itself. Not affiliated with Homebrew or Y Combinator.</p>
      </footer>
    </>
  );
}
