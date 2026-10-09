"""Score trained logo models against each other, the same way for every run.

  fd       Frechet distance between samples and the real marks, in the feature space
           of a small autoencoder trained on the marks (lower is better). Same idea as
           FID, but Inception was trained on photos and these are 48px ink masks.
  copies   share of samples that are a near-copy of one training mark (lower is better).
  sane     share of samples with between 3% and 70% ink: not blank, not a blob.
  ms       milliseconds per mark on CPU through onnxruntime (proxy for the browser).

    python model/logo_eval.py flow-w48-s12000 ddpm-w48-s12000 --steps 8 16 32 --guidance 1 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent))
from logo_train import RUNS, UNet, load_data, sample, save_grid  # noqa: E402

COUNT = 1000


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(),        # 24
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),       # 12
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),      # 6
            nn.Flatten(), nn.Linear(128 * 36, 128),
        )

    def forward(self, x):
        return self.net(x)


def feature_encoder(images: torch.Tensor) -> Encoder:
    """Train (once) an autoencoder on the real marks and keep its encoder."""
    path = RUNS / "features.pt"
    encoder = Encoder()
    if path.exists():
        encoder.load_state_dict(torch.load(path))
        return encoder.eval()
    decoder = nn.Sequential(
        nn.Linear(128, 128 * 36), nn.Unflatten(1, (128, 6, 6)), nn.SiLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
        nn.ConvTranspose2d(32, 1, 4, 2, 1),
    )
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(params, lr=1e-3)
    torch.manual_seed(0)
    for step in range(3000):
        batch = images[torch.randint(0, len(images), (128,))]
        loss = F.mse_loss(decoder(encoder(batch)), batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 500 == 0:
            print(f"  feature autoencoder step {step} loss {loss.item():.4f}")
    torch.save(encoder.state_dict(), path)
    return encoder.eval()


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(cov_a @ cov_b).real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * root))


def nearest_distance(samples: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Mean absolute pixel difference (0..2) from each sample to its closest real mark."""
    s = samples.flatten(1)
    best = torch.full((len(s),), 9.0)
    for chunk in real.flatten(1).split(1000):
        best = torch.minimum(best, torch.cdist(s, chunk, p=1).min(1).values / s.shape[1])
    return best


def onnx_ms(net: UNet, guidance: float) -> float:
    """CPU time of one network call (two with guidance) via onnxruntime."""
    import onnxruntime as ort
    path = RUNS / "timing.onnx"
    torch.onnx.export(
        net, (torch.zeros(1, 1, 48, 48), torch.zeros(1), torch.zeros(1, dtype=torch.long)),
        path, input_names=["x", "t", "label"], output_names=["out"], dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {"x": np.zeros((1, 1, 48, 48), np.float32), "t": np.zeros(1, np.float32),
            "label": np.zeros(1, np.int64)}
    for _ in range(3):
        session.run(None, feed)
    started = time.time()
    for _ in range(20):
        session.run(None, feed)
    calls = 1 if guidance == 1 else 2
    return (time.time() - started) / 20 * 1000 * calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0])
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    images, labels = load_data("cpu")
    encoder = feature_encoder(images)
    brands = images[labels == 0]
    with torch.no_grad():
        real_features = encoder(brands).numpy()
        half = len(brands) // 2
        floor = frechet(real_features[:half], real_features[half:])
    print(f"floor (brand marks, half vs half): fd {floor:.2f}")
    print(f"{'run':24s} {'steps':>5s} {'cfg':>4s} {'fd':>7s} {'copies':>7s} {'sane':>6s} {'ms/mark':>8s}")

    for run in args.runs:
        saved = torch.load(RUNS / run / "ema.pt")
        net = UNet(saved["width"])
        net.load_state_dict(saved["state"])
        net.eval()
        call_ms = onnx_ms(net, 1.0)
        net = net.to(device)
        for steps in args.steps:
            for guidance in args.guidance:
                torch.manual_seed(7)
                noise = torch.randn(COUNT, 1, 48, 48)
                label = torch.zeros(COUNT, dtype=torch.long)       # ask for brand marks
                out = torch.cat([
                    sample(net, saved["method"], n.to(device), l.to(device), steps, guidance).cpu()
                    for n, l in zip(noise.split(250), label.split(250))
                ])
                with torch.no_grad():
                    fd = frechet(encoder(out).numpy(), real_features)
                copies = (nearest_distance(out, images) < 0.05).float().mean().item()
                ink = ((out + 1) / 2).mean((1, 2, 3))
                sane = ((ink > 0.03) & (ink < 0.7)).float().mean().item()
                ms = call_ms * steps * (1 if guidance == 1 else 2)
                print(f"{run:24s} {steps:5d} {guidance:4.1f} {fd:7.2f} {copies:7.1%} {sane:6.1%} {ms:8.0f}")
                save_grid(out[:64], RUNS / run / f"eval-s{steps}-g{guidance:g}.png")


if __name__ == "__main__":
    main()
