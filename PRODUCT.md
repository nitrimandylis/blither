# blither

Names for your next CLI, app or startup, from a 49k-parameter character model
that runs in the browser. Pick what you are naming, pick how bold, get a name
plus a live check of whether it is taken on npm, Homebrew, PyPI, crates.io or
as a .com/.dev/.ai.

Inspired by yname (github.com/aadithyanr/yname), which does this for YC names
only. blither trains on three corpora at once and conditions on which one you
want: Homebrew formulae (cli), Homebrew casks (app), the YC directory (startup).

## Where it is

- Model trained on the filtered corpus (no lib*, digits, hyphens), exported, running in the Vite app.
- Hints: up to four words about the product; the batch is two hint+word-part names
  scored by the model and three model spins, hint-started ones ranked first.
  Style picks the temperatures and which word parts are allowed.
- Acceptance test passed 2026-09-13: eight real names replayed from their descriptions.
- History at /history: kept names and every batch, localStorage. Dark mode toggle.
  Alpino + Plex Mono self-hosted. Category and style remembered.
- Live at https://blither.vercel.app since 2026-09-13, redeploys on push to main.

## Where it is headed

- Maybe: `renamed` state on the name-it button; a per-name page. Neither asked for.
- Rejected: npm as a corpus, it is more hyphenated than Homebrew. Filtering fixed the shape.
- Not doing: an LLM, a backend, accounts, a GitHub availability check, `any` category.

## Retrain

```
python3 data/build_corpus.py
uv run --with numpy model/train.py
```

Overfits past ~4,000 steps on this corpus; the default stops there.
