// bun src/lib/mark.check.ts — the same name always gives the same mark, and marks vary.
import { drawMark } from "./mark";

const check = (ok: boolean, what: string) => {
  if (!ok) throw new Error(what);
  console.log("ok", what);
};

check(drawMark("blither", 0) === drawMark("blither", 0), "same name and variant, same mark");
check(drawMark("blither", 0) !== drawMark("blither", 1), "another variant, another mark");

const names = ["blither", "quorra", "fenwick", "tidal", "oxbow", "mirth", "kestrel", "plinth", "sorrel", "vantage"];
const marks = new Set(names.map((n) => drawMark(n, 0)));
check(marks.size >= 9, "ten names give at least nine different marks");

const path = drawMark("blither", 0);
check(/^M/.test(path) && path.split("M").length - 1 === 4 && !/NaN/.test(path), "one closed shape per cell, no NaN");
for (const n of path.match(/-?\d+(\.\d+)?/g)!.map(Number)) check(n >= 0 && n <= 48, `coordinate ${n} inside the 48 box`);
