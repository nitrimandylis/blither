"""Train a small U-Net that draws 48x48 logo marks, two ways, so they can be compared.

    --method ddpm   denoising diffusion (Ho et al. 2020): the net predicts the noise
                    that was added; sampling walks back with DDIM steps.
    --method flow   flow matching / rectified flow (Lipman et al. 2022, Liu et al. 2022):
                    the net predicts the straight-line velocity from image to noise;
                    sampling follows it back with Euler steps.

Same network, data, batch and step count for both; only the training target and
the sampler differ. The net also takes a style label (0 brand mark, 1 UI icon,
2 none) and drops it 10% of the time so classifier-free guidance works.

    python model/logo_train.py --method flow --steps 20000
    -> model/runs/<name>/ema.pt, log.csv, grid.png
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "model/runs"
NO_LABEL = 2


# ---------------------------------------------------------------- network

def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1], like the position encoding in transformers."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = 1000 * t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(e)[:, :, None, None]          # tell the block how noisy x is
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Self-attention over the 12x12 grid, so far-apart parts of a mark agree."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        weights = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        y = (v @ weights.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.out(y)


class UNet(nn.Module):
    """48 -> 24 -> 12 and back up, with skip connections between matching sizes."""

    def __init__(self, width: int):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 2
        emb = width * 4
        self.width = width
        self.time_mlp = nn.Sequential(nn.Linear(width, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.label = nn.Embedding(3, emb)
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.down1 = ResBlock(c1, c1, emb)
        self.down2 = ResBlock(c1, c2, emb)
        self.down3 = ResBlock(c2, c3, emb)
        self.mid1 = ResBlock(c3, c3, emb)
        self.mid_attn = Attention(c3)
        self.mid2 = ResBlock(c3, c3, emb)
        self.up3 = ResBlock(c3 + c3, c2, emb)
        self.up2 = ResBlock(c2 + c2, c1, emb)
        self.up1 = ResBlock(c1 + c1, c1, emb)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t, label):
        e = self.time_mlp(time_embedding(t, self.width)) + self.label(label)
        h1 = self.down1(self.inp(x), e)                          # 48
        h2 = self.down2(F.avg_pool2d(h1, 2), e)                  # 24
        h3 = self.down3(F.avg_pool2d(h2, 2), e)                  # 12
        m = self.mid2(self.mid_attn(self.mid1(h3, e)), e)
        u3 = self.up3(torch.cat([m, h3], 1), e)                  # 12
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2.0), h2], 1), e)   # 24
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2.0), h1], 1), e)   # 48
        return self.out(u1)


# ---------------------------------------------------------------- the two methods

def ddpm_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021): how much image is left at time t."""
    s = 0.008
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


def training_target(method: str, x0, noise, t):
    """Returns (noisy input, what the net should predict)."""
    if method == "ddpm":
        a = ddpm_alpha_bar(t)[:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise
    # flow: a straight line from the image (t=0) to pure noise (t=1)
    tt = t[:, None, None, None]
    return (1 - tt) * x0 + tt * noise, noise - x0


def guided(net, x, t, label, guidance):
    """Classifier-free guidance: push away from the unlabelled prediction."""
    with_label = net(x, t, label)
    if guidance == 1.0:
        return with_label
    without = net(x, t, torch.full_like(label, NO_LABEL))
    return without + guidance * (with_label - without)


@torch.no_grad()
def sample(net, method, noise, label, steps, guidance=1.0):
    """Turn noise into images in `steps` network calls (twice that with guidance)."""
    x = noise
    times = torch.linspace(1, 0, steps + 1, device=x.device)
    if method == "ddpm":
        times = times.clamp(max=0.999)                 # alpha_bar(1) is 0: avoid dividing by it
    for i in range(steps):
        t_now = times[i].expand(x.shape[0])
        t_next = times[i + 1].expand(x.shape[0])
        pred = guided(net, x, t_now, label, guidance)
        if method == "ddpm":
            # DDIM (Song et al. 2020): estimate the clean image, re-noise it to the next level.
            a_now = ddpm_alpha_bar(t_now)[:, None, None, None]
            a_next = ddpm_alpha_bar(t_next)[:, None, None, None]
            x0 = ((x - (1 - a_now).sqrt() * pred) / a_now.sqrt()).clamp(-1, 1)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * pred
        else:
            dt = (t_next - t_now)[:, None, None, None]
            x = x + dt * pred
    return x.clamp(-1, 1)


# ---------------------------------------------------------------- data and loop

def load_data(device):
    d = np.load(ROOT / "data/logos.npz")
    images = torch.tensor(d["images"], dtype=torch.float32)[:, None] / 127.5 - 1   # ink = +1
    labels = torch.tensor(d["source"], dtype=torch.long)
    return images.to(device), labels.to(device)


def augment(x):
    """Random mirror and a shift of up to 2px, so 5.8k marks go further."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    dx, dy = np.random.randint(-2, 3, size=2)
    return torch.roll(F.pad(x, (2, 2, 2, 2), value=-1), (int(dy), int(dx)), (2, 3))[:, :, 2:-2, 2:-2]


def save_grid(images: torch.Tensor, path: Path, columns: int = 8):
    pixels = ((images[:, 0].clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    rows = len(pixels) // columns
    grid = pixels[: rows * columns].reshape(rows, columns, 48, 48).transpose(0, 2, 1, 3)
    grid = 255 - grid.reshape(rows * 48, columns * 48)
    Image.fromarray(grid).resize((columns * 96, rows * 96), Image.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ddpm", "flow"], required=True)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    name = args.name or f"{args.method}-w{args.width}-s{args.steps}"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    images, labels = load_data(device)
    net = UNet(args.width).to(device)
    ema = UNet(args.width).to(device)
    ema.load_state_dict(net.state_dict())
    print(f"{name}: {sum(p.numel() for p in net.parameters()):,} params, {len(images)} images, {device}")
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "loss", "seconds"])
    started = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        pick = torch.randint(0, len(images), (args.batch,), device=device)
        x0 = augment(images[pick])
        label = torch.where(torch.rand(args.batch, device=device) < 0.1, NO_LABEL, labels[pick])
        t = torch.rand(args.batch, device=device)
        noise = torch.randn_like(x0)
        noisy, target = training_target(args.method, x0, noise, t)
        loss = F.mse_loss(net(noisy, t, label), target)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # Warm up the learning rate over the first 500 steps, then cosine-decay it.
        warm = min(1.0, step / 500)
        decay = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimiser.param_groups:
            group["lr"] = args.lr * warm * (0.1 + 0.9 * decay)
        optimiser.step()

        # Exponential moving average of the weights: smoother samples than the raw net.
        with torch.no_grad():
            for e_p, p in zip(ema.parameters(), net.parameters()):
                e_p.lerp_(p, 1 - 0.999)

        running += loss.item() if step % 50 == 0 else 0.0
        if step % 1000 == 0:
            seconds = time.time() - started
            writer.writerow([step, round(running / 20, 5), round(seconds)])
            log.flush()
            print(f"step {step:6d}  loss {running / 20:.4f}  {seconds:.0f}s")
            running = 0.0

    torch.save({"method": args.method, "width": args.width, "state": ema.state_dict()}, out / "ema.pt")
    torch.manual_seed(1)
    noise = torch.randn(64, 1, 48, 48, device=device)
    label = torch.zeros(64, dtype=torch.long, device=device)
    save_grid(sample(ema.eval(), args.method, noise, label, steps=32), out / "grid.png")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
