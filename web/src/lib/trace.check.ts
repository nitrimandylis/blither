// bun src/lib/trace.check.ts — traces shapes with known answers.
import { area, traceLoops, traceToPath } from "./trace";

function image(inside: (x: number, y: number) => boolean): Float32Array {
  const ink = new Float32Array(48 * 48);
  for (let y = 0; y < 48; y++) for (let x = 0; x < 48; x++) ink[y * 48 + x] = inside(x, y) ? 1 : 0;
  return ink;
}
const distance = (x: number, y: number) => Math.hypot(x - 23.5, y - 23.5);
const check = (ok: boolean, what: string) => {
  if (!ok) throw new Error(what);
  console.log("ok", what);
};

const disc = traceLoops(image((x, y) => distance(x, y) < 15), 48);
check(disc.length === 1, "a disc is one loop");
check(Math.abs(Math.abs(area(disc[0])) - Math.PI * 15 * 15) < 40, `disc area near pi r^2 (got ${Math.abs(area(disc[0])).toFixed(0)})`);

const ring = traceLoops(image((x, y) => distance(x, y) < 15 && distance(x, y) > 8), 48);
check(ring.length === 2, "a ring is two loops");
check(Math.sign(area(ring[0])) !== Math.sign(area(ring[1])), "the hole winds the other way");

const two = traceLoops(image((x, y) => (x < 15 || x > 30) && y > 10 && y < 30), 48);
check(two.length === 2, "two blobs are two loops");

const speck = traceLoops(image((x, y) => x === 5 && y === 5), 48);
check(speck.length === 0, "a single-pixel speck is dropped");

// every loop must close: a broken orientation would leave short open chains
const checker = traceLoops(image((x, y) => (x + y) % 2 === 0 && x > 10 && x < 20 && y > 10 && y < 20), 48);
check(checker.every((l) => l.length >= 3), "checkerboard saddles still give closed loops");

check(/^M.*Z$/.test(traceToPath(image((x, y) => distance(x, y) < 10))), "path is M ... Z");
