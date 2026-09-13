# blither

Names for your next CLI, app or startup, from a 49k-parameter character model
that runs in the browser. Pick what you are naming, pick how bold, get a name
plus a live check of whether it is taken on npm, Homebrew, PyPI, crates.io or
as a .com/.dev/.ai.

Inspired by yname (github.com/aadithyanr/yname), which does this for YC names
only. blither trains on three corpora at once and conditions on which one you
want: Homebrew formulae (cli), Homebrew casks (app), the YC directory (startup).

## Where it is

- Model trained, exported, and running in the Vite app. Seeds reproduce batches.
- Not yet on GitHub or Vercel.

## Where it is headed

- Deploy to Vercel (root directory: `web`).
- Maybe: more corpora (npm top packages, PyPI) if the cli names feel too Homebrew-shaped.
- Not doing: an LLM, a backend, accounts, saved names.

## Retrain

```
python3 data/build_corpus.py
uv run --with numpy model/train.py
```

Overfits past ~4,000 steps on this corpus; the default stops there.
