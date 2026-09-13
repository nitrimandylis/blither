import { forward, type Model } from "./model";

export type Style = "sensible" | "bold" | "unhinged";

// Sampling temperatures per style. Higher = stranger letters.
const TEMPERATURES: Record<Style, number[]> = {
  sensible: [0.66, 0.72, 0.78, 0.84],
  bold: [0.72, 0.84, 0.96, 1.08],
  unhinged: [0.88, 1.0, 1.12, 1.24],
};

const ATTEMPTS = 200;

// Word parts that real product names glue onto a hint: token+pilot, grid+cast, drone+hub.
const SUFFIXES = ["pilot", "cast", "pack", "hub", "kit", "base", "flow", "stack", "lab", "ly", "ify", "io", "sync",
  "scope", "sense", "mind", "wise", "path", "lane", "loop", "nest", "wave", "shift", "spark", "forge", "craft",
  "desk", "board", "box", "bit", "byte", "dock", "port", "gate", "link", "mate", "pal", "bot", "ai", "os", "ware",
  "line", "note", "log", "map", "beam", "core", "grid", "deck", "drop", "leaf", "root", "seed", "well", "guard"];
const PREFIXES = ["open", "auto", "quick", "true", "deep", "clear", "meta", "neo", "re", "co", "my", "go", "up", "pro"];
const FILLER = new Set(["ai", "labs", "health", "app", "cli", "io", "tech", "bio", "technologies", "systems"]);
const BLOCKED = ["fuck", "shit", "bitch", "cunt", "dick", "cock", "pussy", "nigg", "fag", "slut", "whore"];

// Deterministic PRNG so a seed in the URL reproduces the batch.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

function sampleToken(logits: Float32Array, temperature: number, blocked: Set<number>, random: () => number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (!blocked.has(i)) max = Math.max(max, logits[i] / temperature);
  }
  const weights = new Float64Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    if (blocked.has(i)) continue;
    weights[i] = Math.exp(logits[i] / temperature - max);
    total += weights[i];
  }
  let cursor = random() * total;
  for (let i = 0; i < weights.length; i++) {
    cursor -= weights[i];
    if (cursor <= 0) return i;
  }
  return weights.length - 1;
}

function logSoftmax(logits: Float32Array, id: number): number {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  let total = 0;
  for (const v of logits) total += Math.exp(v - max);
  return logits[id] - max - Math.log(total);
}

// Returns the name and how likely the model found it (mean log-prob per character).
// With a prefix, the model continues from those letters instead of starting blank.
function sampleName(model: Model, categoryId: number, temperature: number, random: () => number, prefix = "") {
  const start = model.tokenToId.get("<START>")!;
  const end = model.tokenToId.get("<END>")!;
  const context: number[] = Array(model.manifest.context).fill(start);
  const chars: string[] = [];
  let logProb = 0;
  for (const ch of prefix) {
    const id = model.tokenToId.get(ch);
    if (id === undefined) continue;
    chars.push(ch);
    context.shift();
    context.push(id);
  }
  for (let step = chars.length; step < 22; step++) {
    const logits = forward(model, context, categoryId);
    const blocked = new Set([start]);
    if (chars.length < 3) blocked.add(end);
    const id = sampleToken(logits, temperature, blocked, random);
    logProb += logSoftmax(logits, id);
    if (id === end) break;
    chars.push(model.manifest.tokens[id]);
    context.shift();
    context.push(id);
  }
  return { name: chars.join("").trim(), logProb: logProb / (chars.length + 1) };
}

function editDistance(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)));
    }
    prev = cur;
  }
  return prev[b.length];
}

function looksUsable(name: string): boolean {
  if (name.length < 3 || name.length > 16) return false;
  if (/(.)\1\1/.test(name)) return false;                  // aaa
  if (/  /.test(name)) return false;
  if ((name.match(/ /g) ?? []).length > 1) return false;
  const letters = name.replace(/[^a-z]/g, "");
  if (letters.length < 3) return false;
  if (letters.length >= 5 && !/[aeiouy]/.test(letters)) return false;
  if (/[bcdfghjklmnpqrstvwxz]{5}/.test(letters)) return false;
  if (BLOCKED.some((bad) => letters.includes(bad))) return false;
  return true;
}

// Reject anything within edit distance 1 of a real name (2 for longer names).
// ponytail: linear scan over ~19k names per candidate, only run on the top few; index it if it ever feels slow
function tooClose(name: string, known: string[]): boolean {
  const limit = name.length <= 6 ? 1 : 2;
  for (const other of known) {
    if (Math.abs(other.length - name.length) > limit) continue;
    if (editDistance(name, other) <= limit) return true;
  }
  return false;
}

export function cleanHints(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter((word) => word.length >= 2)
    .slice(0, 4);
}

// How likely the model finds an existing string: mean log-prob per character, teacher-forced.
function scoreName(model: Model, categoryId: number, name: string): number {
  const start = model.tokenToId.get("<START>")!;
  const end = model.tokenToId.get("<END>")!;
  const context: number[] = Array(model.manifest.context).fill(start);
  let logProb = 0;
  const ids = [...name].map((ch) => model.tokenToId.get(ch) ?? start).concat(end);
  for (const id of ids) {
    logProb += logSoftmax(forward(model, context, categoryId), id);
    context.shift();
    context.push(id);
  }
  return logProb / ids.length;
}

// tokens -> token, quizzes -> quiz, drone -> dron (for vowel-led suffixes)
function stems(hint: string): string[] {
  let word = hint;
  if (word.length > 4 && word.endsWith("es")) word = word.slice(0, -2);
  else if (word.length > 3 && word.endsWith("s")) word = word.slice(0, -1);
  word = word.replace(/([b-df-hj-np-tv-z])\1$/, "$1");
  const bare = /[aeiouy]$/.test(word) ? word.slice(0, -1) : word;
  return [...new Set([word, bare])];
}

// hint glued to a word part. Suffix forms first; a prefix form is marked so it can rank lower,
// since "openfoo" and "profoo" are generic
function composed(hints: string[]): Array<{ name: string; prefixed: boolean }> {
  const out = new Map<string, boolean>();
  for (const hint of hints) {
    const [word, bare] = stems(hint);
    for (const suffix of SUFFIXES) {
      out.set(word + suffix, false);
      if (/^[aeiouy]/.test(suffix)) out.set(bare + suffix, false);
    }
    for (const prefix of PREFIXES) out.set(prefix + word, true);
  }
  return [...out.entries()].map(([name, prefixed]) => ({ name, prefixed }));
}

function sharesTrigram(name: string, hints: string[]): boolean {
  const flat = name.replace(/ /g, "");
  for (const hint of hints) {
    for (let i = 0; i + 3 <= hint.length; i++) {
      if (flat.includes(hint.slice(i, i + 3))) return true;
    }
  }
  return false;
}

export function generate(model: Model, category: string, style: Style, count: number, seed: number, hints: string[] = []): string[] {
  const random = mulberry32(seed);
  const categories = model.manifest.categories;
  const allKnown = Object.values(model.known).flat();
  const temps = TEMPERATURES[style];
  // three sources, ranked separately then interleaved, so a hinted batch always carries the hints
  const pools = { glued: new Map<string, number>(), completed: new Map<string, number>(), wheel: new Map<string, number>() };

  function consider(pool: Map<string, number>, name: string, logProb: number, bonus = 0) {
    if (pool.has(name) || !looksUsable(name)) return;
    // a misspelling or plural of the hint is not a name: "drona", "videos", "fatige"
    if (hints.some((hint) => editDistance(name, hint) <= 2)) return;
    const words = name.split(" ");
    const filler = words.filter((w) => FILLER.has(w)).length;
    // prefer names the model finds likely, around 7 chars, one word, no filler
    pool.set(name, logProb + bonus - Math.abs(name.length - 7) * 0.03 - (words.length - 1) * 0.08 - filler * 0.25);
  }

  // hint completions: the model continues from the first 2..5 letters of each hint word,
  // never the whole word, so it has to contribute something
  for (const hint of hints) {
    for (let len = 3; len <= Math.min(5, hint.length - 1); len++) {
      for (let round = 0; round < temps.length * 3; round++) {
        const chosen = category === "any" ? categories[Math.floor(random() * categories.length)] : category;
        const { name, logProb } = sampleName(model, categories.indexOf(chosen), temps[round % temps.length], random, hint.slice(0, len));
        if (name.length - len < 3) continue;
        consider(pools.completed, name, logProb);
      }
    }
  }

  // hint + word part, ranked by how name-like the model finds the result
  for (const { name, prefixed } of composed(hints)) {
    const chosen = category === "any" ? categories[Math.floor(random() * categories.length)] : category;
    // a little seeded noise so "another" reshuffles the glued names too
    consider(pools.glued, name, scoreName(model, categories.indexOf(chosen), name), (prefixed ? -0.3 : 0) + random() * 0.3);
  }

  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    // "any" picks a random category per name so the batch mixes flavours
    const chosen = category === "any" ? categories[Math.floor(random() * categories.length)] : category;
    const { name, logProb } = sampleName(model, categories.indexOf(chosen), temps[attempt % temps.length], random);
    consider(pools.wheel, name, logProb, hints.length && sharesTrigram(name, hints) ? 0.25 : 0);
  }
  const rank = (pool: Map<string, number>) => [...pool.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name);
  const order = hints.length ? [pools.glued, pools.completed, pools.glued, pools.glued, pools.wheel] : [pools.wheel];
  const ranked = order.map(rank);
  const results: string[] = [];
  const perHint = new Map<string, number>();
  // walk the sources in turn; the known-name scan is the slow part, so it only runs on names that get this far
  for (let i = 0; results.length < count; i++) {
    const list = ranked[i % ranked.length];
    let name: string | undefined;
    while ((name = list.shift()) !== undefined) {
      // at most two names built on the same hint, so a batch is not five takes on "fatigue"
      const hint = hints.find((h) => name!.startsWith(h.slice(0, 2)));
      if (hint && (perHint.get(hint) ?? 0) >= 2) continue;
      if (results.includes(name) || tooClose(name, allKnown)) continue;
      if (hint) perHint.set(hint, (perHint.get(hint) ?? 0) + 1);
      results.push(name);
      break;
    }
    if (ranked.every((l) => l.length === 0)) break;
  }
  return results;
}
