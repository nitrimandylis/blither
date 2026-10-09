// bun src/lib/store.check.ts — pick and mark survive a trip through the URL.
import { batchUrl, readCount } from "./store";

const check = (ok: boolean, what: string) => {
  if (!ok) throw new Error(what);
  console.log("ok", what);
};

const base = { category: "cli", style: "bold", seed: 42, hints: "" };
check(batchUrl(base) === "/?for=cli&style=bold&seed=42", "pick and mark 0 leave the old link shape alone");

const params = new URLSearchParams(batchUrl({ ...base, pick: 3, mark: 7 }).slice(1));
check(readCount(params, "pick") === 3 && readCount(params, "mark") === 7, "pick and mark round-trip");

for (const junk of ["-2", "1.5", "abc", ""]) {
  check(readCount(new URLSearchParams(`mark=${junk}`), "mark") === 0, `junk mark "${junk}" reads as 0`);
}
