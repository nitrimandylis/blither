"""Train a character-level MLP on data/corpus.csv and export it for the browser.

Architecture (Bengio et al. 2003, sized like yname):

    previous CONTEXT characters -> character embeddings
    + category embedding
    -> tanh hidden layer
    -> next-character logits (+ a per-category bias)

Outputs web/public/model/manifest.json, weights.bin, known.json.

    uv run --with numpy model/train.py
"""

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "web/public/model"
START, END = "<START>", "<END>"

CONTEXT = 10
EMBED = 24
HIDDEN = 160
CAT_EMBED = 12


def load_corpus(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as f:
        return [(row["name"], row["category"]) for row in csv.DictReader(f)]


def make_examples(rows, tok, cat):
    """Every (previous CONTEXT chars, category) -> next char, incl. END."""
    xs, ys, cs = [], [], []
    for name, category in rows:
        context = [tok[START]] * CONTEXT
        for ch in list(name) + [END]:
            xs.append(context.copy())
            ys.append(tok[ch])
            cs.append(cat[category])
            context = context[1:] + [tok[ch]]
    return np.array(xs, np.int32), np.array(ys, np.int32), np.array(cs, np.int32)


def init_params(rng, vocab, cats):
    inp = CONTEXT * EMBED + CAT_EMBED
    return {
        "char_embedding": rng.normal(0, 0.1, (vocab, EMBED)),
        "category_embedding": rng.normal(0, 0.1, (cats, CAT_EMBED)),
        "w1": rng.normal(0, np.sqrt(2 / inp), (inp, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "w2": rng.normal(0, 0.02, (HIDDEN, vocab)),
        "b2": np.zeros(vocab),
        "category_output": np.zeros((cats, vocab)),
    }


def forward(p, xs, cs):
    chars = p["char_embedding"][xs].reshape(len(xs), -1)
    features = np.concatenate([chars, p["category_embedding"][cs]], axis=1)
    hidden = np.tanh(features @ p["w1"] + p["b1"])
    logits = hidden @ p["w2"] + p["b2"] + p["category_output"][cs]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return features, hidden, probs


def loss_and_grads(p, xs, ys, cs):
    features, hidden, probs = forward(p, xs, cs)
    n = len(xs)
    loss = -np.mean(np.log(probs[np.arange(n), ys] + 1e-12))

    d_logits = probs
    d_logits[np.arange(n), ys] -= 1
    d_logits /= n

    g = {}
    g["w2"] = hidden.T @ d_logits
    g["b2"] = d_logits.sum(axis=0)
    g["category_output"] = np.zeros_like(p["category_output"])
    np.add.at(g["category_output"], cs, d_logits)

    d_hidden = (d_logits @ p["w2"].T) * (1 - hidden**2)
    g["w1"] = features.T @ d_hidden
    g["b1"] = d_hidden.sum(axis=0)

    d_features = d_hidden @ p["w1"].T
    d_chars = d_features[:, : CONTEXT * EMBED].reshape(n, CONTEXT, EMBED)
    g["char_embedding"] = np.zeros_like(p["char_embedding"])
    np.add.at(g["char_embedding"], xs, d_chars)
    g["category_embedding"] = np.zeros_like(p["category_embedding"])
    np.add.at(g["category_embedding"], cs, d_features[:, CONTEXT * EMBED :])
    return loss, g


def adam_step(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
        p[k] -= lr * (m[k] / (1 - b1**t)) / (np.sqrt(v[k] / (1 - b2**t)) + eps)


def val_loss(p, xs, ys, cs):
    _, _, probs = forward(p, xs, cs)
    return float(-np.mean(np.log(probs[np.arange(len(xs)), ys] + 1e-12)))


def sample(p, tokens, tok, category_id, rng, temperature=0.8):
    context = [tok[START]] * CONTEXT
    out = []
    for _ in range(MAX_SAMPLE := 22):
        _, _, probs = forward(p, np.array([context]), np.array([category_id]))
        logits = np.log(probs[0] + 1e-12) / temperature
        logits[tok[START]] = -1e9
        if len(out) < 3:
            logits[tok[END]] = -1e9
        logits -= logits.max()
        w = np.exp(logits)
        next_id = int(rng.choice(len(tokens), p=w / w.sum()))
        if next_id == tok[END]:
            break
        out.append(tokens[next_id])
        context = context[1:] + [next_id]
    return "".join(out)


def export(p, tokens, categories, rows, stats):
    OUT.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    meta = {}
    for k, arr in p.items():
        flat = arr.astype(np.float32).ravel()
        meta[k] = {"shape": list(arr.shape), "offset": len(blob) // 4, "length": flat.size}
        blob += flat.tobytes()
    (OUT / "weights.bin").write_bytes(blob)
    (OUT / "manifest.json").write_text(json.dumps({
        "context": CONTEXT, "embed": EMBED, "hidden": HIDDEN, "catEmbed": CAT_EMBED,
        "tokens": tokens, "categories": categories, "params": meta, "training": stats,
    }))
    known = {}
    for name, category in rows:
        known.setdefault(category, []).append(name)
    (OUT / "known.json").write_text(json.dumps(known))
    print(f"exported {len(blob)} bytes of weights to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load_corpus(ROOT / "data/corpus.csv")
    tokens = [START, END] + sorted({ch for name, _ in rows for ch in name})
    tok = {t: i for i, t in enumerate(tokens)}
    categories = sorted({c for _, c in rows})
    cat = {c: i for i, c in enumerate(categories)}

    order = rng.permutation(len(rows))
    n_val = len(rows) // 10
    val_rows = [rows[i] for i in order[:n_val]]
    train_rows = [rows[i] for i in order[n_val:]]
    train = make_examples(train_rows, tok, cat)
    val = make_examples(val_rows, tok, cat)
    print(f"{len(rows)} names, {len(train[0])} train examples, {len(val[0])} val examples")

    p = init_params(rng, len(tokens), len(categories))
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    best, best_params, t0 = float("inf"), None, time.time()

    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train[0]), args.batch)
        lr = args.lr if step < args.steps * 0.7 else args.lr / 4
        loss, g = loss_and_grads(p, train[0][idx], train[1][idx], train[2][idx])
        adam_step(p, g, m, v, step, lr)
        if step % 250 == 0 or step == args.steps:
            vl = val_loss(p, *val)
            flag = ""
            if vl < best:
                best, best_params, flag = vl, {k: x.copy() for k, x in p.items()}, " *"
            print(f"step {step:5d}  train {loss:.3f}  val {vl:.3f}  {time.time() - t0:.0f}s{flag}")

    print(f"\nbest val loss {best:.3f} (perplexity {np.exp(best):.1f})")
    for c in categories:
        names = [sample(best_params, tokens, tok, cat[c], rng) for _ in range(12)]
        print(f"{c:8} {', '.join(names)}")
    export(best_params, tokens, categories, rows, {"names": len(rows), "valLoss": best})


if __name__ == "__main__":
    main()
