import { useEffect, useState } from "react";
import { batchUrl, store } from "./lib/store";

function when(at: number): string {
  const days = Math.floor((Date.now() - at) / 86_400_000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  return `${days} days ago`;
}

export default function History() {
  const [kept, setKept] = useState(store.kept);
  const batches = store.batches();

  useEffect(() => {
    document.title = "history, blither";
  }, []);

  function drop(name: string) {
    store.unkeep(name);
    setKept(store.kept());
  }

  return (
    <>
      <section className="kept">
        <h2 className="lead">kept</h2>
        {kept.length === 0 ? (
          <p className="muted">Nothing yet. Press keep under a name you might use.</p>
        ) : (
          <ul>
            {kept.map((k) => (
              <li key={k.name}>
                <a className="mono" href={batchUrl(k)}>{k.name}</a>
                <span className="muted">{k.category}{k.hints ? `, ${k.hints}` : ""}</span>
                <button type="button" className="link" onClick={() => drop(k.name)}>drop</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="batches">
        <h2 className="lead">every batch, newest first, this browser only</h2>
        {batches.length === 0 ? (
          <p className="muted">No batches yet.</p>
        ) : (
          <ul>
            {batches.map((b) => (
              <li key={`${b.seed}-${b.hints}-${b.category}`}>
                <a href={batchUrl(b)} className="mono">{b.names.join("  ")}</a>
                <span className="muted">{b.category}, {b.style}{b.hints ? `, ${b.hints}` : ""}, {when(b.at)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
