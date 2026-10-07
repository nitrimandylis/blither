"""Export one trained logo run for the browser.

Writes web/public/model/logo.onnx (the U-Net) and logo.json (method, steps,
guidance), then checks that sampling through onnxruntime, written the way
web/src/lib/logo.ts does it, gives the same mark as sample() in PyTorch.

    python model/logo_export.py flow-w48-s12000 --steps 16 --guidance 2
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
from logo_train import NO_LABEL, RUNS, UNet, sample  # noqa: E402

OUT = Path(__file__).parent.parent / "web/public/model"


def alpha_bar(t: float) -> float:
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def sample_like_the_browser(session, method, noise, steps, guidance):
    """Line for line what drawMark() in logo.ts does, in NumPy."""
    x = noise.reshape(-1).astype(np.float32).copy()
    for i in range(steps):
        t_now, t_next = 1 - i / steps, 1 - (i + 1) / steps
        if method == "ddpm":
            t_now, t_next = min(t_now, 0.999), min(t_next, 0.999)
        batch = 1 if guidance == 1 else 2
        out = session.run(None, {
            "x": np.stack([x, x][:batch]).reshape(batch, 1, 48, 48),
            "t": np.full(batch, t_now, np.float32),
            "label": np.array([0, NO_LABEL][:batch], np.int64),
        })[0].reshape(batch, -1)
        pred = out[0] if batch == 1 else out[1] + guidance * (out[0] - out[1])
        if method == "ddpm":
            a_now, a_next = alpha_bar(t_now), alpha_bar(t_next)
            x0 = np.clip((x - math.sqrt(1 - a_now) * pred) / math.sqrt(a_now), -1, 1)
            x = math.sqrt(a_next) * x0 + math.sqrt(1 - a_next) * pred
        else:
            x = x + (t_next - t_now) * pred
    return np.clip(x, -1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args()

    saved = torch.load(RUNS / args.run / "ema.pt")
    net = UNet(saved["width"])
    net.load_state_dict(saved["state"])
    net.eval()

    onnx_path = OUT / "logo.onnx"
    torch.onnx.export(
        net, (torch.zeros(2, 1, 48, 48), torch.zeros(2), torch.zeros(2, dtype=torch.long)),
        onnx_path, input_names=["x", "t", "label"], output_names=["out"],
        dynamic_axes={"x": {0: "batch"}, "t": {0: "batch"}, "label": {0: "batch"}, "out": {0: "batch"}},
        dynamo=False,
    )
    manifest = {"method": saved["method"], "steps": args.steps, "guidance": args.guidance, "run": args.run}
    (OUT / "logo.json").write_text(json.dumps(manifest, indent=2) + "\n")

    torch.manual_seed(3)
    noise = torch.randn(1, 1, 48, 48)
    expected = sample(net, saved["method"], noise, torch.zeros(1, dtype=torch.long), args.steps, args.guidance)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sample_like_the_browser(session, saved["method"], noise.numpy(), args.steps, args.guidance)
    gap = np.abs(got - expected.numpy().reshape(-1)).max()
    assert gap < 1e-3, f"browser-style sampling drifts from PyTorch by {gap}"
    print(f"exported {args.run} ({onnx_path.stat().st_size / 1e6:.1f} MB), parity gap {gap:.1e}")


if __name__ == "__main__":
    main()
