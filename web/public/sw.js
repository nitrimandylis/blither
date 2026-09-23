// Offline support: every same-origin GET goes to the network first and the
// answer is saved; with no network the saved copy is served instead.
// Network first (not cache first) so a retrained model or a new deploy
// reaches the phone the next time it is online.
const CACHE = "blither-v1";

// Fetched at install so the app works offline even before the model has
// been used once. The hashed JS and CSS get saved on the first page load.
const SHELL = [
  "/",
  "/model/manifest.json",
  "/model/weights.bin",
  "/model/known.json",
  "/fonts/alpino-400.woff2",
  "/fonts/alpino-500.woff2",
  "/fonts/plex-mono-400.woff2",
  "/fonts/plex-mono-500.woff2",
  "/favicon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // drop caches from older versions of this file
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key)))),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  // registry and domain checks are other origins: leave them alone
  if (request.method !== "GET" || new URL(request.url).origin !== location.origin) return;
  event.respondWith(fromNetworkOrCache(request));
});

async function fromNetworkOrCache(request) {
  const cache = await caches.open(CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch {
    const saved = await cache.match(request);
    if (saved) return saved;
    // /history and any other page is the same single-page app
    if (request.mode === "navigate") return cache.match("/");
    return Response.error();
  }
}
