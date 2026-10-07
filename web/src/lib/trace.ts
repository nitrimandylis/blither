// Turns a 48x48 ink image into an SVG path, so a drawn mark is crisp at any size.
//
// 1. Marching squares: walk every 2x2 block of pixels and find where the ink
//    crosses 50%, interpolating between pixels so outlines are not stair-stepped.
// 2. Join those little segments into closed loops (outer edges and holes).
// 3. Drop specks, smooth each loop a little, and draw it with curves.
// Holes come out right because the path is filled with the even-odd rule.

const THRESHOLD = 0.5;
const MIN_AREA = 1.5;   // loops smaller than this many square pixels are speckle

type Point = { x: number; y: number };

// The image with a one-pixel border of paper, so every outline closes.
function padded(ink: Float32Array, size: number): { value: (x: number, y: number) => number; size: number } {
  const big = size + 2;
  const grid = new Float32Array(big * big);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) grid[(y + 1) * big + (x + 1)] = ink[y * size + x];
  }
  return { value: (x, y) => grid[y * big + x], size: big };
}

export function traceLoops(ink: Float32Array, size: number): Point[][] {
  const { value, size: big } = padded(ink, size);

  // Where the outline crosses each pixel edge. Keys name the edge, so two cells
  // that share an edge produce the same key and their segments join up.
  function crossing(key: string, ax: number, ay: number, bx: number, by: number): { key: string; point: Point } {
    const a = value(ax, ay);
    const b = value(bx, by);
    const t = (THRESHOLD - a) / (b - a);
    return { key, point: { x: ax + (bx - ax) * t, y: ay + (by - ay) * t } };
  }

  // start edge -> (end edge, start point); every segment keeps ink on its left
  const next = new Map<string, { to: string; point: Point }>();

  for (let y = 0; y < big - 1; y++) {
    for (let x = 0; x < big - 1; x++) {
      const tl = value(x, y) >= THRESHOLD;
      const tr = value(x + 1, y) >= THRESHOLD;
      const br = value(x + 1, y + 1) >= THRESHOLD;
      const bl = value(x, y + 1) >= THRESHOLD;
      const top = tl !== tr ? crossing(`h${x},${y}`, x, y, x + 1, y) : null;
      const right = tr !== br ? crossing(`v${x + 1},${y}`, x + 1, y, x + 1, y + 1) : null;
      const bottom = bl !== br ? crossing(`h${x},${y + 1}`, x, y + 1, x + 1, y + 1) : null;
      const left = tl !== bl ? crossing(`v${x},${y}`, x, y, x, y + 1) : null;

      let pairs: Array<[NonNullable<typeof top>, NonNullable<typeof top>]> = [];
      const crossed = [top, right, bottom, left].filter((c) => c !== null);
      if (crossed.length === 2) {
        pairs = [[crossed[0], crossed[1]]];
      } else if (crossed.length === 4) {
        // Saddle: two opposite corners are ink. The centre decides whether they connect.
        const centreInk = (value(x, y) + value(x + 1, y) + value(x + 1, y + 1) + value(x, y + 1)) / 4 >= THRESHOLD;
        const cutTopRightAndBottomLeft = tl === centreInk;
        pairs = cutTopRightAndBottomLeft ? [[top!, right!], [bottom!, left!]] : [[left!, top!], [right!, bottom!]];
      }

      const corners: Record<string, { x: number; y: number; ink: boolean }> = {
        tl: { x, y, ink: tl }, tr: { x: x + 1, y, ink: tr },
        br: { x: x + 1, y: y + 1, ink: br }, bl: { x, y: y + 1, ink: bl },
      };
      for (let [a, b] of pairs) {
        // A corner that is not on the segment tells us which side the ink is on:
        // the corner the segment cuts off, or any corner when it runs straight across.
        const sides = [a.key === top?.key || b.key === top?.key, a.key === right?.key || b.key === right?.key,
          a.key === bottom?.key || b.key === bottom?.key, a.key === left?.key || b.key === left?.key];
        const [t, r, bo, l] = sides;
        const corner = t && r ? corners.tr : r && bo ? corners.br : bo && l ? corners.bl : corners.tl;
        // cross product > 0: the corner is on the left of a -> b (y pointing down)
        const cross = (b.point.x - a.point.x) * (corner.y - a.point.y) - (b.point.y - a.point.y) * (corner.x - a.point.x);
        const inkOnLeft = cross > 0 ? corner.ink : !corner.ink;
        if (!inkOnLeft) [a, b] = [b, a];
        next.set(a.key, { to: b.key, point: a.point });
      }
    }
  }

  // Follow segments end to start until each loop closes.
  const loops: Point[][] = [];
  const used = new Set<string>();
  for (const start of next.keys()) {
    if (used.has(start)) continue;
    const loop: Point[] = [];
    let key: string | undefined = start;
    while (key !== undefined && !used.has(key)) {
      used.add(key);
      const step: { to: string; point: Point } | undefined = next.get(key);
      if (!step) break;
      loop.push({ x: step.point.x - 1, y: step.point.y - 1 });   // undo the border
      key = step.to;
    }
    if (loop.length >= 3 && Math.abs(area(loop)) >= MIN_AREA) loops.push(loop);
  }
  return loops;
}

// Shoelace formula: signed area of a closed polygon.
export function area(loop: Point[]): number {
  let sum = 0;
  for (let i = 0; i < loop.length; i++) {
    const a = loop[i];
    const b = loop[(i + 1) % loop.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return sum / 2;
}

// One pass of neighbour averaging takes the last of the pixel steps out.
function smooth(loop: Point[]): Point[] {
  return loop.map((p, i) => {
    const before = loop[(i - 1 + loop.length) % loop.length];
    const after = loop[(i + 1) % loop.length];
    return { x: (before.x + 2 * p.x + after.x) / 4, y: (before.y + 2 * p.y + after.y) / 4 };
  });
}

const round = (n: number) => Math.round(n * 100) / 100;

// Catmull-Rom through the points, written as cubic Bezier curves.
function loopToPath(loop: Point[]): string {
  const n = loop.length;
  const at = (i: number) => loop[(i + n) % n];
  let d = `M${round(loop[0].x)} ${round(loop[0].y)}`;
  for (let i = 0; i < n; i++) {
    const p0 = at(i - 1);
    const p1 = at(i);
    const p2 = at(i + 1);
    const p3 = at(i + 2);
    const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
    const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
    d += `C${round(c1.x)} ${round(c1.y)} ${round(c2.x)} ${round(c2.y)} ${round(p2.x)} ${round(p2.y)}`;
  }
  return d + "Z";
}

// The full path for a 48x48 ink image, in a 0..48 viewBox. Fill it with fill-rule="evenodd".
export function traceToPath(ink: Float32Array, size = 48): string {
  return traceLoops(ink, size).map((loop) => loopToPath(smooth(loop))).join("");
}
