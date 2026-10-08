import { useEffect, type CSSProperties } from "react";
import { drawTiles, saveMark } from "./lib/mark";

const TYPE_MS = 38;   // per character, the same pace as useTyped in App.tsx

// The mark as four tiles. With build on, the tiles appear one by one over the
// time the name takes to type.
export function Mark({ name, variant, build = false }: { name: string; variant: number; build?: boolean }) {
  const total = name.length * TYPE_MS;
  return (
    <svg
      key={`${name}#${variant}`}   // a new key restarts the build animation
      className={build ? "mark building" : "mark"}
      viewBox="0 0 48 48"
      role="img"
      aria-label={`mark ${variant + 1} for ${name}`}
      style={{ "--build": `${total}ms` } as CSSProperties}
    >
      {drawTiles(name, variant).map((d, i) => (
        <path key={i} d={d} fill="currentColor" style={{ "--tile": i } as CSSProperties} />
      ))}
    </svg>
  );
}

// ‹ mark 4 › and save svg. The arrow keys step too, unless focus is in a text box.
export function MarkControls({ name, variant, setVariant }: { name: string; variant: number; setVariant: (v: number) => void }) {
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "ArrowRight") setVariant(variant + 1);
      if (event.key === "ArrowLeft" && variant > 0) setVariant(variant - 1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [variant, setVariant]);

  return (
    <div className="mark-controls">
      <div className="stepper">
        <button type="button" className="step" onClick={() => setVariant(variant - 1)} disabled={variant === 0} aria-label="previous mark">‹</button>
        <span aria-live="polite">mark {variant + 1}</span>
        <button type="button" className="step" onClick={() => setVariant(variant + 1)} aria-label="next mark">›</button>
      </div>
      <button type="button" className="link" onClick={() => saveMark(name, variant)}>save svg</button>
    </div>
  );
}
