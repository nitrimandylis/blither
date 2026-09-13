import { forward, type Model } from "./model";

export type Style = "sensible" | "bold" | "unhinged";

// Sampling temperatures per style. Higher = stranger letters.
const TEMPERATURES: Record<Style, number[]> = {
  sensible: [0.66, 0.72, 0.78, 0.84],
  bold: [0.72, 0.84, 0.96, 1.08],
  unhinged: [0.88, 1.0, 1.12, 1.24],
};

const ATTEMPTS = 200;
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
function sampleName(model: Model, categoryId: number, temperature: number, random: () => number) {
  const start = model.tokenToId.get("<START>")!;
  const end = model.tokenToId.get("<END>")!;
  const context: number[] = Array(model.manifest.context).fill(start);
  const chars: string[] = [];
  let logProb = 0;
  for (let step = 0; step < 22; step++) {
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
  if (!/^[a-z0-9]/.test(name) || !/[a-z0-9]$/.test(name)) return false;
  if (/(.)\1\1/.test(name)) return false;                  // aaa
  if (/[^a-z0-9]{2}/.test(name)) return false;              // "--", " -"
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

export function generate(model: Model, category: string, style: Style, count: number, seed: number): string[] {
  const random = mulberry32(seed);
  const categories = model.manifest.categories;
  const allKnown = Object.values(model.known).flat();
  const temps = TEMPERATURES[style];
  const scored = new Map<string, number>();

  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    // "any" picks a random category per name so the batch mixes flavours
    const chosen = category === "any" ? categories[Math.floor(random() * categories.length)] : category;
    const { name, logProb } = sampleName(model, categories.indexOf(chosen), temps[attempt % temps.length], random);
    if (scored.has(name) || !looksUsable(name)) continue;
    const words = name.split(/[ -]/);
    const filler = words.filter((w) => FILLER.has(w)).length;
    // prefer names the model finds likely, around 7 chars, one word, no filler
    scored.set(name, logProb - Math.abs(name.length - 7) * 0.02 - (words.length - 1) * 0.08 - filler * 0.25);
  }
  // the known-name scan is the slow part, so only run it on the best candidates
  const ranked = [...scored.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name);
  const results: string[] = [];
  for (const name of ranked) {
    if (results.length === count) break;
    if (!tooClose(name, allKnown)) results.push(name);
  }
  return results;
}
