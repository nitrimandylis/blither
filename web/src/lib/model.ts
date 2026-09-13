// Loads the exported weights and runs one forward pass of the character MLP.
// Mirrors forward() in model/train.py exactly.

type ParamMeta = { shape: number[]; offset: number; length: number };

export type Manifest = {
  context: number;
  embed: number;
  hidden: number;
  catEmbed: number;
  tokens: string[];
  categories: string[];
  params: Record<string, ParamMeta>;
  training: { names: number; valLoss: number };
};

export type Model = {
  manifest: Manifest;
  params: Record<string, Float32Array>;
  tokenToId: Map<string, number>;
  known: Record<string, string[]>;
};

export async function loadModel(): Promise<Model> {
  const [manifest, weights, known] = await Promise.all([
    fetch("/model/manifest.json").then((r) => r.json() as Promise<Manifest>),
    fetch("/model/weights.bin").then((r) => r.arrayBuffer()),
    fetch("/model/known.json").then((r) => r.json() as Promise<Record<string, string[]>>),
  ]);
  const params: Record<string, Float32Array> = {};
  for (const [name, meta] of Object.entries(manifest.params)) {
    params[name] = new Float32Array(weights, meta.offset * 4, meta.length);
  }
  const tokenToId = new Map(manifest.tokens.map((token, id) => [token, id]));
  return { manifest, params, tokenToId, known };
}

// context: the previous `context` token ids. Returns raw logits, one per token.
export function forward(model: Model, context: number[], categoryId: number): Float32Array {
  const { context: contextSize, embed, hidden: hiddenSize, catEmbed, tokens } = model.manifest;
  const { char_embedding, category_embedding, w1, b1, w2, b2, category_output } = model.params;
  const vocab = tokens.length;

  // 1. build the input vector: every context char's embedding, then the category's
  const inputSize = contextSize * embed + catEmbed;
  const features = new Float32Array(inputSize);
  for (let pos = 0; pos < contextSize; pos++) {
    for (let d = 0; d < embed; d++) {
      features[pos * embed + d] = char_embedding[context[pos] * embed + d];
    }
  }
  for (let d = 0; d < catEmbed; d++) {
    features[contextSize * embed + d] = category_embedding[categoryId * catEmbed + d];
  }

  // 2. hidden = tanh(features @ w1 + b1)
  const hidden = new Float64Array(hiddenSize);
  for (let h = 0; h < hiddenSize; h++) hidden[h] = b1[h];
  for (let i = 0; i < inputSize; i++) {
    const x = features[i];
    for (let h = 0; h < hiddenSize; h++) hidden[h] += x * w1[i * hiddenSize + h];
  }
  for (let h = 0; h < hiddenSize; h++) hidden[h] = Math.tanh(hidden[h]);

  // 3. logits = hidden @ w2 + b2 + category_output[category]
  const logits = new Float32Array(vocab);
  for (let o = 0; o < vocab; o++) logits[o] = b2[o] + category_output[categoryId * vocab + o];
  for (let h = 0; h < hiddenSize; h++) {
    const x = hidden[h];
    for (let o = 0; o < vocab; o++) logits[o] += x * w2[h * vocab + o];
  }
  return logits;
}
