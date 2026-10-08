// Draws a 48x48 logo mark for a name: a 2x2 grid of geometric tiles (quarter
// circles, half circles, leaves, triangles), usually arranged with rotational
// symmetry so it reads as one mark. The same name and variant always give the
// same mark.

import { mulberry32 } from "./generate";

const SIZE = 48;
const GAP = 2;
const CELL = (SIZE - GAP) / 2;   // 23
const HALF = CELL / 2;

// A tile is drawn in its own cell, from (0, 0) to (CELL, CELL), before rotation.
type Command = ["M" | "L", number, number] | ["A", number, number, number, number] | ["Z"];

const TILES: Command[][] = [
  // quarter circle, centred on the top-left corner
  [["M", 0, 0], ["L", CELL, 0], ["A", CELL, 1, 0, CELL], ["Z"]],
  // square with a quarter circle bitten out of the top-left corner
  [["M", CELL, 0], ["L", CELL, CELL], ["L", 0, CELL], ["A", CELL, 0, CELL, 0], ["Z"]],
  // half circle hanging from the top edge
  [["M", 0, 0], ["L", CELL, 0], ["A", HALF, 1, 0, 0], ["Z"]],
  // leaf: two quarter arcs between opposite corners
  [["M", 0, CELL], ["A", CELL, 1, CELL, 0], ["A", CELL, 1, 0, CELL], ["Z"]],
  // triangle: the top-right half of the cell
  [["M", 0, 0], ["L", CELL, 0], ["L", CELL, CELL], ["Z"]],
  // full circle
  [["M", 0, HALF], ["A", HALF, 1, CELL, HALF], ["A", HALF, 1, 0, HALF], ["Z"]],
  // full square
  [["M", 0, 0], ["L", CELL, 0], ["L", CELL, CELL], ["L", 0, CELL], ["Z"]],
];

// Top-left corner of each cell, clockwise from the top-left.
const CELLS = [
  [0, 0],
  [CELL + GAP, 0],
  [CELL + GAP, CELL + GAP],
  [0, CELL + GAP],
];

// Rotates a point inside the cell by quarter turns clockwise, around the cell's centre.
function rotate(x: number, y: number, turns: number): [number, number] {
  for (let i = 0; i < turns % 4; i++) {
    const dx = x - HALF;
    const dy = y - HALF;
    x = HALF - dy;
    y = HALF + dx;
  }
  return [x, y];
}

function round(n: number): string {
  return String(Math.round(n * 100) / 100);
}

function tilePath(tile: Command[], cell: number, turns: number): string {
  const [left, top] = CELLS[cell];
  const parts: string[] = [];
  for (const command of tile) {
    if (command[0] === "Z") {
      parts.push("Z");
    } else if (command[0] === "A") {
      // rotation keeps an arc's radius and sweep direction, only its end point moves
      const [, radius, sweep, x, y] = command;
      const [rx, ry] = rotate(x, y, turns);
      parts.push(`A${round(radius)} ${round(radius)} 0 0 ${sweep} ${round(left + rx)} ${round(top + ry)}`);
    } else {
      const [letter, x, y] = command;
      const [rx, ry] = rotate(x, y, turns);
      parts.push(`${letter}${round(left + rx)} ${round(top + ry)}`);
    }
  }
  return parts.join("");
}

// FNV-1a: turns the name into a 32-bit seed.
function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// One SVG path per tile, clockwise from the top-left, so the page can build them in one by one.
export function drawTiles(name: string, variant: number): string[] {
  const random = mulberry32(hash(`${name}#${variant}`));
  const pick = (n: number) => Math.floor(random() * n);

  const tiles: number[] = [];
  const turns: number[] = [];
  const style = random();
  if (style < 0.45) {
    // pinwheel: one tile, turned a quarter more in each cell
    const tile = pick(TILES.length - 2);   // a full circle or square would look the same in every cell
    const start = pick(4);
    for (let cell = 0; cell < 4; cell++) {
      tiles.push(tile);
      turns.push(start + cell);
    }
  } else if (style < 0.85) {
    // pair: two tiles, each repeated in the opposite cell turned half a turn
    const a = pick(TILES.length);
    // two plain tiles (circle, square) make a dull mark, four squares is the Windows logo
    const b = a >= TILES.length - 2 ? pick(TILES.length - 2) : pick(TILES.length);
    const turnA = pick(4);
    const turnB = pick(4);
    tiles.push(a, b, a, b);
    turns.push(turnA, turnB, turnA + 2, turnB + 2);
  } else {
    // free: four unrelated tiles
    for (let cell = 0; cell < 4; cell++) {
      tiles.push(pick(TILES.length));
      turns.push(pick(4));
    }
  }

  const paths: string[] = [];
  for (let cell = 0; cell < 4; cell++) paths.push(tilePath(TILES[tiles[cell]], cell, turns[cell]));
  return paths;
}

// The whole mark as one path.
export function drawMark(name: string, variant: number): string {
  return drawTiles(name, variant).join("");
}

// Downloads the mark as name.svg. currentColor is black on its own and takes the
// text colour when the file is inlined into a page.
export function saveMark(name: string, variant: number) {
  const file = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="currentColor" d="${drawMark(name, variant)}"/></svg>\n`;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([file], { type: "image/svg+xml" }));
  link.download = `${name}.svg`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);   // Safari cancels the download if revoked at once
}
