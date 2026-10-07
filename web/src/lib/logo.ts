// Draws a 48x48 logo mark for a name with the U-Net from model/logo_train.py.
// The network runs through onnxruntime-web; the sampling loop is here and mirrors
// sample() in logo_train.py. The same name and variant always give the same mark.

import type { InferenceSession } from "onnxruntime-web";
import { mulberry32 } from "./generate";

const SIZE = 48;
const BRAND = 0;
const NO_LABEL = 2;

type LogoManifest = { method: "ddpm" | "flow"; steps: number; guidance: number };
type LogoModel = { session: InferenceSession; manifest: LogoManifest; ort: typeof import("onnxruntime-web/wasm") };

let loading: Promise<LogoModel> | null = null;

// The runtime is ~3.5 MB compressed, so it is only fetched the first time a mark is drawn.
function loadLogoModel(): Promise<LogoModel> {
  if (!loading) {
    loading = (async () => {
      const ort = await import("onnxruntime-web/wasm");
      const [manifest, session] = await Promise.all([
        fetch("/model/logo.json").then((r) => r.json() as Promise<LogoManifest>),
        ort.InferenceSession.create("/model/logo.onnx"),
      ]);
      return { session, manifest, ort };
    })();
  }
  return loading;
}

// FNV-1a: turns the name into a 32-bit seed.
function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Standard normal noise from the seeded PRNG (Box-Muller).
function gaussianNoise(random: () => number): Float32Array {
  const noise = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < noise.length; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    noise[i] = r * Math.cos(angle);
    noise[i + 1] = r * Math.sin(angle);
  }
  return noise;
}

function alphaBar(t: number): number {
  const s = 0.008;
  return Math.cos(((t + s) / (1 + s)) * (Math.PI / 2)) ** 2;
}

// Returns ink per pixel, 0 (paper) to 1 (ink), row by row.
export async function drawMark(name: string, variant: number): Promise<Float32Array> {
  const { session, manifest, ort } = await loadLogoModel();
  const { method, steps, guidance } = manifest;
  const x = gaussianNoise(mulberry32(hash(`${name}#${variant}`)));
  const pixels = SIZE * SIZE;

  for (let i = 0; i < steps; i++) {
    let tNow = 1 - i / steps;
    let tNext = 1 - (i + 1) / steps;
    if (method === "ddpm") {
      tNow = Math.min(tNow, 0.999);   // alpha_bar(1) is 0: avoid dividing by it
      tNext = Math.min(tNext, 0.999);
    }

    // With guidance, one call with a batch of two: with the brand label and without.
    const batch = guidance === 1 ? 1 : 2;
    const input = new Float32Array(pixels * batch);
    const times = new Float32Array(batch).fill(tNow);
    const labels = new BigInt64Array(batch).fill(BigInt(BRAND));
    input.set(x, 0);
    if (batch === 2) {
      input.set(x, pixels);
      labels[1] = BigInt(NO_LABEL);
    }
    const result = await session.run({
      x: new ort.Tensor("float32", input, [batch, 1, SIZE, SIZE]),
      t: new ort.Tensor("float32", times, [batch]),
      label: new ort.Tensor("int64", labels, [batch]),
    });
    const out = result.out.data as Float32Array;

    for (let p = 0; p < pixels; p++) {
      const withLabel = out[p];
      const pred = batch === 1 ? withLabel : out[pixels + p] + guidance * (withLabel - out[pixels + p]);
      if (method === "ddpm") {
        // DDIM: estimate the clean image, then re-noise it to the next level
        const aNow = alphaBar(tNow);
        const aNext = alphaBar(tNext);
        const x0 = Math.max(-1, Math.min(1, (x[p] - Math.sqrt(1 - aNow) * pred) / Math.sqrt(aNow)));
        x[p] = Math.sqrt(aNext) * x0 + Math.sqrt(1 - aNext) * pred;
      } else {
        x[p] = x[p] + (tNext - tNow) * pred;
      }
    }
  }

  const ink = new Float32Array(pixels);
  for (let p = 0; p < pixels; p++) ink[p] = Math.max(0, Math.min(1, (x[p] + 1) / 2));
  return ink;
}
