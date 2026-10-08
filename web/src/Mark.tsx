import { useEffect, useState } from "react";
import { drawMark } from "./lib/mark";

function svgFile(path: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill-rule="evenodd" d="${path}"/></svg>\n`;
}

export default function Mark({ name }: { name: string }) {
  const [variant, setVariant] = useState(0);

  useEffect(() => setVariant(0), [name]);

  if (!name) return <div className="mark-row" />;
  const path = drawMark(name, variant);

  function save() {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([svgFile(path)], { type: "image/svg+xml" }));
    link.download = `${name}.svg`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);   // Safari cancels the download if revoked at once
  }

  return (
    <div className="mark-row">
      <svg className="mark" viewBox="0 0 48 48" role="img" aria-label={`generated mark for ${name}`}>
        <path fill="currentColor" fillRule="evenodd" d={path} />
      </svg>
      <button type="button" className="link" onClick={() => setVariant((v) => v + 1)}>another mark</button>
      <button type="button" className="link" onClick={save}>save svg</button>
    </div>
  );
}
