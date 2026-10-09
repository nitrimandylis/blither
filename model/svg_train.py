"""Train a small GPT that draws logo marks as SVG path tokens (IconShop-style, Wu et al. 2023).

Same job as the character model, one level up: there it is "given the letters so
far, which letter next?", here "given the pen commands so far, which token next?".
Tokens come from data/build_svg_tokens.py.

Sampling is masked by the path grammar (after C come exactly six coordinates, L/C/Z
only inside a shape, ...) so every sample is a valid path; the network still decides
everything the grammar allows.

    python model/svg_train.py --steps 10000
    -> model/runs/svg-<...>/model.pt, log.csv, grid.png
"""

import argparse
import csv
import io
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data"))
from build_svg_tokens import (  # noqa: E402
    CLOSE, COORD, CURVE, END, GRID, LABEL, LINE, MAX_LEN, MOVE, PAD, VOCAB, to_path,
)

RUNS = ROOT / "model/runs"
COORDS_AFTER = {MOVE: 2, LINE: 2, CURVE: 6, CLOSE: 0}


# ---------------------------------------------------------------- network

class Block(nn.Module):
    """One transformer layer: masked self-attention, then a two-layer MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, past_k=None, past_v=None):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(c, dim=2)
        # (batch, heads, time, head_dim)
        q, k, v = (z.reshape(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        if past_k is not None:
            # generating: earlier tokens' keys and values are cached, only the new token is computed
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            y = F.scaled_dot_product_attention(q, k, v)        # one new token sees everything before it
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        x = x + self.mlp(self.norm2(x))
        return x, k, v


class PathGPT(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int):
        super().__init__()
        self.dim, self.layers_count, self.heads = dim, layers, heads
        self.token = nn.Embedding(VOCAB, dim)
        self.position = nn.Embedding(MAX_LEN, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB, bias=False)

    def forward(self, tokens):
        """Training: logits for every position at once."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        for block in self.blocks:
            x, _, _ = block(x)
        return self.head(self.norm(x))


class PathGPTStep(nn.Module):
    """Generation: one new token in, using the cached keys/values of all earlier ones.

    cache is (layers, 2, batch, heads, time, head_dim); the browser passes it back in
    on every call. This is the module that gets exported to ONNX.
    """

    def __init__(self, gpt: PathGPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, token, position, cache):
        x = self.gpt.token(token) + self.gpt.position(position)[:, None, :]
        new_keys, new_values = [], []
        for i, block in enumerate(self.gpt.blocks):
            x, k, v = block(x, cache[i, 0], cache[i, 1])
            new_keys.append(k)
            new_values.append(v)
        logits = self.gpt.head(self.gpt.norm(x))[:, -1]
        new_cache = torch.stack([torch.stack([k, v]) for k, v in zip(new_keys, new_values)])
        return logits, new_cache


# ---------------------------------------------------------------- grammar

class Grammar:
    """Tracks where we are in the path so impossible tokens can be masked out."""

    def __init__(self):
        self.coords_left = 0
        self.in_shape = False
        self.commands = 0

    def allowed(self) -> list[int]:
        if self.coords_left > 0:
            return list(range(COORD, COORD + GRID))
        options = [MOVE]
        if self.in_shape:
            options += [LINE, CURVE, CLOSE]
        if self.commands > 0 and not self.in_shape:
            options.append(END)          # only stop after a closed shape
        return options

    def push(self, token: int):
        if token >= COORD:
            self.coords_left -= 1
        elif token in COORDS_AFTER:
            self.coords_left = COORDS_AFTER[token]
            self.commands += 1
            if token == MOVE:
                self.in_shape = True
            if token == CLOSE:
                self.in_shape = False


@torch.no_grad()
def sample_paths(gpt: PathGPT, count: int, label: int, temperature: float, seed: int, device) -> list[list[int]]:
    """Draw `count` paths at once, token by token, with the KV cache."""
    step = PathGPTStep(gpt).eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    head_dim = gpt.dim // gpt.heads
    cache = torch.zeros(gpt.layers_count, 2, count, gpt.heads, 0, head_dim, device=device)
    token = torch.full((count, 1), LABEL + label, dtype=torch.long, device=device)
    grammars = [Grammar() for _ in range(count)]
    paths = [[] for _ in range(count)]
    done = [False] * count
    for position in range(MAX_LEN - 1):
        logits, cache = step(token, torch.full((count,), position, device=device), cache)
        logits = logits.float().cpu() / temperature
        mask = torch.full_like(logits, float("-inf"))
        for i, grammar in enumerate(grammars):
            mask[i, grammar.allowed() if not done[i] else [PAD]] = 0
            if position == MAX_LEN - 2 and not done[i]:
                mask[i] = float("-inf")
                mask[i, END if not grammar.in_shape and grammar.coords_left == 0 else CLOSE] = 0
        choice = torch.multinomial(torch.softmax(logits + mask, dim=1), 1, generator=generator)[:, 0]
        for i in range(count):
            if done[i]:
                continue
            t = int(choice[i])
            if t == END:
                done[i] = True
            else:
                grammars[i].push(t)
                paths[i].append(t)
        if all(done):
            break
        token = choice[:, None].to(device)
    return paths


# ---------------------------------------------------------------- rendering

def render(tokens: list[int], size: int = 48, mark: int = 40) -> np.ndarray:
    """Tokens -> ink mask laid out exactly like data/logos.npz (40px mark on a 48px canvas)."""
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {GRID - 1} {GRID - 1}"><path d="{to_path(tokens)}"/></svg>'
    png = subprocess.run(["rsvg-convert", "-w", str(mark), "-h", str(mark)], input=svg.encode(), capture_output=True).stdout
    canvas = np.zeros((size, size), np.uint8)
    if png:
        alpha = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[:, :, 3]
        offset = (size - mark) // 2
        canvas[offset:offset + mark, offset:offset + mark] = alpha
    return canvas


def save_grid(paths: list[list[int]], path: Path, columns: int = 8):
    tiles = [255 - render(p, 96, 80) for p in paths]
    rows = len(tiles) // columns
    grid = np.array(tiles[: rows * columns]).reshape(rows, columns, 96, 96).transpose(0, 2, 1, 3)
    Image.fromarray(grid.reshape(rows * 96, columns * 96).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- data and loop

def augment(tokens: torch.Tensor) -> torch.Tensor:
    """Mirror half the batch left-right by flipping every x coordinate."""
    tokens = tokens.clone()
    is_coord = tokens >= COORD
    # x is the first of each coordinate pair: count coords along the row, even ones are x
    order = torch.cumsum(is_coord.long(), dim=1) - 1
    is_x = is_coord & (order % 2 == 0)
    flip = (torch.rand(tokens.shape[0], 1, device=tokens.device) < 0.5) & is_x
    tokens[flip] = COORD + (GRID - 1) - (tokens[flip] - COORD)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    name = args.name or f"svg-d{args.dim}-l{args.layers}-s{args.steps}"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(ROOT / "data/svg_tokens.npz")
    tokens = torch.tensor(data["tokens"].astype(np.int64), device=device)
    # hold out 5% to see when it starts memorising
    order = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(0))
    held = order[: len(tokens) // 20].to(device)
    train = order[len(tokens) // 20:].to(device)

    gpt = PathGPT(args.dim, args.layers, args.heads).to(device)
    print(f"{name}: {sum(p.numel() for p in gpt.parameters()):,} params, {len(train)} train paths, {device}", flush=True)
    optimiser = torch.optim.AdamW(gpt.parameters(), lr=args.lr, weight_decay=0.1)

    def loss_on(batch):
        logits = gpt(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1), ignore_index=PAD)

    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train_loss", "held_loss", "seconds"])
    started, running, best = time.time(), 0.0, float("inf")
    for step in range(1, args.steps + 1):
        batch = augment(tokens[train[torch.randint(0, len(train), (args.batch,), device=device)]])
        loss = loss_on(batch)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        warm = min(1.0, step / 500)
        decay = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimiser.param_groups:
            group["lr"] = args.lr * warm * (0.1 + 0.9 * decay)
        optimiser.step()
        running += loss.item() if step % 50 == 0 else 0.0

        if step % 500 == 0:
            gpt.eval()
            with torch.no_grad():
                held_loss = sum(loss_on(tokens[chunk]).item() for chunk in held.split(64)) / len(held.split(64))
            gpt.train()
            if held_loss < best:            # keep the checkpoint from before it starts memorising
                best = held_loss
                torch.save({"dim": args.dim, "layers": args.layers, "heads": args.heads, "state": gpt.state_dict()}, out / "model.pt")
            seconds = time.time() - started
            writer.writerow([step, round(running / 10, 4), round(held_loss, 4), round(seconds)])
            log.flush()
            print(f"step {step:6d}  train {running / 10:.4f}  held {held_loss:.4f}  {seconds:.0f}s", flush=True)
            running = 0.0

    saved = torch.load(out / "model.pt")
    gpt.load_state_dict(saved["state"])
    save_grid(sample_paths(gpt.eval(), 64, 0, 1.0, seed=1, device=device), out / "grid.png")
    print(f"saved {out} (best held loss {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
