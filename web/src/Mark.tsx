import { useEffect, useState } from "react";
import { drawMark } from "./lib/logo";
import { store } from "./lib/store";

// Ink amounts -> a PNG whose alpha is the ink, used as a CSS mask so the mark
// takes the current ink colour and follows dark mode without redrawing.
function inkToPng(ink: Float32Array): string {
  const canvas = document.createElement("canvas");
  canvas.width = 48;
  canvas.height = 48;
  const context = canvas.getContext("2d")!;
  const image = context.createImageData(48, 48);
  for (let p = 0; p < ink.length; p++) image.data[p * 4 + 3] = Math.round(ink[p] * 255);
  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

// The logo model is opt-in: its runtime is a few MB, so nothing loads until asked.
export default function Mark({ name }: { name: string }) {
  const [enabled, setEnabled] = useState(() => store.prefs().marks === "on");
  const [variant, setVariant] = useState(0);
  const [png, setPng] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => setVariant(0), [name]);

  useEffect(() => {
    if (!enabled || !name) return;
    let cancelled = false;
    setPng(null);
    setFailed(false);
    drawMark(name, variant)
      .then((ink) => {
        if (!cancelled) setPng(inkToPng(ink));
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, name, variant]);

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
      <div className={`mark${png ? "" : " drawing"}`} role="img" aria-label={png ? `generated mark for ${name}` : "drawing a mark"}
        style={png ? { maskImage: `url(${png})`, WebkitMaskImage: `url(${png})` } : undefined} />
      {failed ? (
        <span className="muted">the mark model did not load</span>
      ) : (
        <button type="button" className="link" onClick={() => setVariant((v) => v + 1)} disabled={!png}>another mark</button>
      )}
    </div>
  );
}
