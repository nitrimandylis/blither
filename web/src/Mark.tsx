import { useEffect, useState } from "react";
import { drawMark } from "./lib/logo";
import { traceToPath } from "./lib/trace";
import { store } from "./lib/store";

function svgFile(path: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill-rule="evenodd" d="${path}"/></svg>\n`;
}

// The logo model is opt-in: its runtime is a few MB, so nothing loads until asked.
export default function Mark({ name }: { name: string }) {
  const [enabled, setEnabled] = useState(() => store.prefs().marks === "on");
  const [variant, setVariant] = useState(0);
  const [path, setPath] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => setVariant(0), [name]);

  useEffect(() => {
    if (!enabled || !name) return;
    let cancelled = false;
    setPath(null);
    setFailed(false);
    drawMark(name, variant)
      .then((ink) => {
        if (!cancelled) setPath(traceToPath(ink));
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, name, variant]);

  function save() {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([svgFile(path!)], { type: "image/svg+xml" }));
    link.download = `${name}.svg`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);   // Safari cancels the download if revoked at once
  }

  function enable() {
    store.setPref("marks", "on");
    setEnabled(true);
  }

  if (!enabled) {
    return (
      <div className="mark-row">
        <button type="button" className="link" onClick={enable} disabled={!name}>draw a mark</button>
      </div>
    );
  }

  return (
    <div className="mark-row">
      {path ? (
        <svg className="mark" viewBox="0 0 48 48" role="img" aria-label={`generated mark for ${name}`}>
          <path fill="currentColor" fillRule="evenodd" d={path} />
        </svg>
      ) : (
        <div className="mark drawing" role="img" aria-label="drawing a mark" />
      )}
      {failed ? (
        <span className="muted">the mark model did not load</span>
      ) : (
        <>
          <button type="button" className="link" onClick={() => setVariant((v) => v + 1)} disabled={!path}>another mark</button>
          <button type="button" className="link" onClick={save} disabled={!path}>save svg</button>
        </>
      )}
    </div>
  );
}
