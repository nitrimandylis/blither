// Is the name already taken? One fetch per registry, all CORS-open, no backend.

export type Status = "free" | "taken" | "unknown";
export type Check = { label: string; status: Status; url: string };

type Registry = { label: string; api: (slug: string) => string; url: (slug: string) => string };

const REGISTRIES: Record<string, Registry> = {
  npm: {
    label: "npm",
    api: (s) => `https://registry.npmjs.org/${s}`,
    url: (s) => `https://www.npmjs.com/package/${s}`,
  },
  brew: {
    label: "brew",
    api: (s) => `https://formulae.brew.sh/api/formula/${s}.json`,
    url: (s) => `https://formulae.brew.sh/formula/${s}`,
  },
  pypi: {
    label: "pypi",
    api: (s) => `https://pypi.org/pypi/${s}/json`,
    url: (s) => `https://pypi.org/project/${s}/`,
  },
  crates: {
    label: "crates.io",
    api: (s) => `https://crates.io/api/v1/crates/${s}`,
    url: (s) => `https://crates.io/crates/${s}`,
  },
  com: {
    label: ".com",
    api: (s) => `https://rdap.verisign.com/com/v1/domain/${s}.com`,
    url: (s) => `https://${s}.com`,
  },
  dev: {
    label: ".dev",
    api: (s) => `https://pubapi.registry.google/rdap/domain/${s}.dev`,
    url: (s) => `https://${s}.dev`,
  },
  app: {
    label: ".app",
    api: (s) => `https://pubapi.registry.google/rdap/domain/${s}.app`,
    url: (s) => `https://${s}.app`,
  },
  ai: {
    label: ".ai",
    api: (s) => `https://rdap.identitydigital.services/rdap/domain/${s}.ai`,
    url: (s) => `https://${s}.ai`,
  },
};

const BY_CATEGORY: Record<string, string[]> = {
  cli: ["npm", "brew", "pypi", "crates"],
  app: ["com", "app", "ai", "dev"],
  startup: ["com", "ai", "dev"],
};

export function slugOf(name: string): string {
  return name.replace(/ /g, "");
}

async function probe(registry: Registry, slug: string): Promise<Check> {
  const base = { label: registry.label, url: registry.url(slug) };
  try {
    const res = await fetch(registry.api(slug), { signal: AbortSignal.timeout(8000) });
    if (res.ok) return { ...base, status: "taken" };
    if (res.status === 404) return { ...base, status: "free" };
    return { ...base, status: "unknown" };
  } catch {
    return { ...base, status: "unknown" };
  }
}

export function checkName(name: string, category: string): Promise<Check[]> {
  const slug = slugOf(name);
  const keys = BY_CATEGORY[category] ?? ["npm", "com"];
  return Promise.all(keys.map((key) => probe(REGISTRIES[key], slug)));
}
