"""Score the SVG token model with the same yardsticks as logo_eval.py.

Samples are rendered to 48x48 exactly like data/logos.npz, so fd and copies are
directly comparable with the diffusion and flow runs.

    python model/svg_eval.py svg-d192-l4-s10000 --temperature 0.8 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
from logo_eval import COUNT, feature_encoder, frechet, nearest_distance  # noqa: E402
from logo_train import RUNS, load_data  # noqa: E402
from svg_train import PathGPT, PathGPTStep, render, sample_paths, save_grid  # noqa: E402


def per_token_ms(gpt: PathGPT) -> float:
    """CPU time for one generated token through onnxruntime, with a cache of 250 tokens."""
    path = RUNS / "timing-svg.onnx"
    head_dim = gpt.dim // gpt.heads
    cache = torch.zeros(gpt.layers_count, 2, 1, gpt.heads, 250, head_dim)
    torch.onnx.export(
        PathGPTStep(gpt).eval(), (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long), cache),
        path, input_names=["token", "position", "cache"], output_names=["logits", "new_cache"],
        dynamic_axes={"cache": {4: "time"}, "new_cache": {4: "time"}}, dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {"token": np.zeros((1, 1), np.int64), "position": np.array([250], np.int64), "cache": cache.numpy()}
    for _ in range(5):
        session.run(None, feed)
    started = time.time()
    for _ in range(50):
        session.run(None, feed)
    return (time.time() - started) / 50 * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.8, 1.0])
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    images, labels = load_data("cpu")
    encoder = feature_encoder(images)
    with torch.no_grad():
        real_features = encoder(images[labels == 0]).numpy()

    saved = torch.load(RUNS / args.run / "model.pt")
    gpt = PathGPT(saved["dim"], saved["layers"], saved["heads"])
    gpt.load_state_dict(saved["state"])
    gpt.eval()
    token_ms = per_token_ms(gpt)
    gpt = gpt.to(device)

    print(f"{'run':24s} {'temp':>5s} {'fd':>7s} {'copies':>7s} {'sane':>6s} {'tokens':>7s} {'ms/mark':>8s}")
    for temperature in args.temperature:
        paths = []
        for chunk in range(COUNT // 250):
            paths += sample_paths(gpt, 250, 0, temperature, seed=7 + chunk, device=device)
        out = torch.tensor(np.stack([render(p) for p in paths]), dtype=torch.float32)[:, None] / 127.5 - 1
        with torch.no_grad():
            fd = frechet(encoder(out).numpy(), real_features)
        copies = (nearest_distance(out, images) < 0.05).float().mean().item()
        ink = ((out + 1) / 2).mean((1, 2, 3))
        sane = ((ink > 0.03) & (ink < 0.7)).float().mean().item()
        length = float(np.mean([len(p) for p in paths]))
        print(f"{args.run:24s} {temperature:5.2f} {fd:7.2f} {copies:7.1%} {sane:6.1%} {length:7.0f} {token_ms * length:8.0f}")
        save_grid(paths[:64], RUNS / args.run / f"eval-t{temperature:g}.png")


if __name__ == "__main__":
    main()
