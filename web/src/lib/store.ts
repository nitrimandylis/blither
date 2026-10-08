// Everything the page remembers lives in localStorage. No accounts, no server.

// pick (which name in the batch) and mark (which variant) are missing on entries kept before marks existed.
export type Kept = { name: string; category: string; style: string; hints: string; seed: number; at: number; pick?: number; mark?: number };
export type Batch = { names: string[]; category: string; style: string; hints: string; seed: number; at: number };

const MAX_BATCHES = 200;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown) {
  localStorage.setItem(key, JSON.stringify(value));
}

export const store = {
  kept: (): Kept[] => read("blither.kept", []),
  keep(entry: Kept) {
    const rest = store.kept().filter((k) => k.name !== entry.name);
    write("blither.kept", [entry, ...rest]);
  },
  unkeep(name: string) {
    write("blither.kept", store.kept().filter((k) => k.name !== name));
  },

  batches: (): Batch[] => read("blither.batches", []),
  logBatch(batch: Batch) {
    const rest = store.batches().filter((b) => b.seed !== batch.seed || b.hints !== batch.hints || b.category !== batch.category);
    write("blither.batches", [batch, ...rest].slice(0, MAX_BATCHES));
  },

  prefs: () => read<{ category?: string; style?: string; theme?: string }>("blither.prefs", {}),
  setPref(key: "category" | "style" | "theme", value: string) {
    write("blither.prefs", { ...store.prefs(), [key]: value });
  },
};

type Place = { category: string; style: string; seed: number; hints: string; pick?: number; mark?: number };

// pick and mark are left out when 0, so links from before marks still read the same.
export function batchUrl(b: Place): string {
  const query = new URLSearchParams({ for: b.category, style: b.style, seed: String(b.seed) });
  if (b.hints) query.set("hints", b.hints);
  if (b.pick) query.set("pick", String(b.pick));
  if (b.mark) query.set("mark", String(b.mark));
  return `/?${query}`;
}

// Reads a non-negative whole number from the URL, 0 if it is missing or junk.
export function readCount(params: URLSearchParams, key: string): number {
  const n = Number(params.get(key));
  return Number.isInteger(n) && n > 0 ? n : 0;
}
