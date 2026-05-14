const CACHE_NAME = 'resynth-ai-cache-v2';
const STATIC_ASSETS = [
  '/static/logo.png',
  '/static/logo.svg',
  '/manifest.json'
];

// Install: cache only static assets, NOT the HTML page itself
self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .catch(() => {})
  );
});

// Activate: delete ALL old caches immediately
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: network-first for HTML/navigation, cache-first for static assets
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Always go to network for HTML pages (never serve cached HTML)
  if (event.request.mode === 'navigate' || event.request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
    return;
  }

  // Cache-first for static assets only
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
