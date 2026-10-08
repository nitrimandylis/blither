import { useEffect, useState } from "react";
import { batchUrl, store, type Kept } from "./lib/store";
import { Mark } from "./Mark";
import { saveMark } from "./lib/mark";

function when(at: number): string {
  const days = Math.floor((Date.now() - at) / 86_400_000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  return `${days} days ago`;
}

// Entries kept before marks existed have no pick: find the name in its logged batch.
function pickOf(k: Kept): number {
  if (k.pick !== undefined) return k.pick;
  const batch = store.batches().find((b) => b.seed === k.seed && b.hints === k.hints && b.category === k.category);
  return Math.max(0, batch?.names.indexOf(k.name) ?? 0);
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
          <ul className="sheet">
            {kept.map((k) => (
              <li key={k.name}>
                <Mark name={k.name} variant={k.mark ?? 0} />
                {/* the name's link covers the whole tile; the two buttons sit above it */}
                <a className="mono" href={batchUrl({ ...k, pick: pickOf(k) })}>{k.name}</a>
                <span className="muted">{k.category}{k.hints ? `, ${k.hints}` : ""}</span>
                <span className="tile-actions">
                  <button type="button" className="link" onClick={() => saveMark(k.name, k.mark ?? 0)}>save svg</button>
                  <button type="button" className="link" onClick={() => drop(k.name)}>drop</button>
                </span>
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
