import { softmax } from "@/lib/numeric";

export const MODEL_CONFIG = {
  layers: 2,
  heads: 2,
  dModel: 8,
  dFf: 16,
  maxTokens: 12,
} as const;

export const VOCAB = [
  "<BOS>", "<EOS>", "<UNK>", "小猫", "小狗", "机器人", "坐", "站", "躺",
  "在", "垫子", "沙发", "草地", "上", "里", "旁边", "的", "是", "很",
  "可爱", "聪明", "安静", "。", "！", "我", "喜欢", "你", "猫", "狗",
] as const;

export type Matrix = number[][];
export type Token = { id: number; text: string; raw: string; unknown: boolean };
export type Candidate = { id: number; token: string; logit: number; probability: number };

export type LayerTrace = {
  input: Matrix;
  normalizedAttention: Matrix;
  queries: Matrix[];
  keys: Matrix[];
  values: Matrix[];
  rawScores: number[][][];
  scores: number[][][];
  attention: number[][][];
  contexts: Matrix[];
  concatenated: Matrix;
  projected: Matrix;
  attentionOutput: Matrix;
  afterAttention: Matrix;
  normalizedMlp: Matrix;
  mlpPreActivation: Matrix;
  mlpHidden: Matrix;
  mlpOutput: Matrix;
  output: Matrix;
};

export type ForwardTrace = {
  tokens: Token[];
  embeddings: Matrix;
  tokenEmbeddings: Matrix;
  positionEmbeddings: Matrix;
  layers: LayerTrace[];
  finalNormalized: Matrix;
  logits: number[];
};

const HEAD_DIM = MODEL_CONFIG.dModel / MODEL_CONFIG.heads;
const tokenToId = new Map<string, number>(VOCAB.map((token, index) => [token, index]));
const sortedVocab = [...VOCAB]
  .filter((token) => !token.startsWith("<"))
  .sort((a, b) => b.length - a.length);

function randomGenerator(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function seededMatrix(rows: number, cols: number, seed: number, scale = 0.55): Matrix {
  const random = randomGenerator(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (random() * 2 - 1) * scale),
  );
}

function add(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]));
}

function multiply(a: Matrix, b: Matrix): Matrix {
  const inner = b.length;
  const cols = b[0].length;
  return a.map((row) =>
    Array.from({ length: cols }, (_, j) => {
      let sum = 0;
      for (let k = 0; k < inner; k += 1) sum += row[k] * b[k][j];
      return sum;
    }),
  );
}

function rmsNorm(input: Matrix): Matrix {
  return input.map((row) => {
    const rms = Math.sqrt(row.reduce((sum, value) => sum + value * value, 0) / row.length + 1e-5);
    return row.map((value, index) => (value / rms) * (0.85 + index * 0.04));
  });
}

function gelu(value: number) {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

const tokenEmbedding = seededMatrix(VOCAB.length, MODEL_CONFIG.dModel, 41, 0.8);
const positionEmbedding = seededMatrix(MODEL_CONFIG.maxTokens, MODEL_CONFIG.dModel, 87, 0.35);

const weights = Array.from({ length: MODEL_CONFIG.layers }, (_, layer) => ({
  wq: Array.from({ length: MODEL_CONFIG.heads }, (_, head) =>
    seededMatrix(MODEL_CONFIG.dModel, HEAD_DIM, 1000 + layer * 100 + head * 11),
  ),
  wk: Array.from({ length: MODEL_CONFIG.heads }, (_, head) =>
    seededMatrix(MODEL_CONFIG.dModel, HEAD_DIM, 1100 + layer * 100 + head * 11),
  ),
  wv: Array.from({ length: MODEL_CONFIG.heads }, (_, head) =>
    seededMatrix(MODEL_CONFIG.dModel, HEAD_DIM, 1200 + layer * 100 + head * 11),
  ),
  wo: seededMatrix(MODEL_CONFIG.dModel, MODEL_CONFIG.dModel, 1300 + layer),
  w1: seededMatrix(MODEL_CONFIG.dModel, MODEL_CONFIG.dFf, 1400 + layer),
  b1: Array.from({ length: MODEL_CONFIG.dFf }, (_, i) => Math.sin(i + layer) * 0.08),
  w2: seededMatrix(MODEL_CONFIG.dFf, MODEL_CONFIG.dModel, 1500 + layer),
  b2: Array.from({ length: MODEL_CONFIG.dModel }, (_, i) => Math.cos(i + layer) * 0.05),
}));

const unembedding = seededMatrix(VOCAB.length, MODEL_CONFIG.dModel, 2048, 0.65);
const outputBias = VOCAB.map((token) => {
  if (token === "。") return 2.4;
  if (token === "<EOS>") return 1.3;
  if (token === "！") return 0.45;
  if (token === "可爱" || token === "安静") return 0.25;
  return 0;
});

export function getLayerParameters(layer: number) {
  return weights[layer];
}

export function getEmbeddingParameters() {
  return { token: tokenEmbedding, position: positionEmbedding };
}

export function getUnembeddingParameters() {
  return { weights: unembedding, bias: outputBias };
}

export function getRmsGains() {
  return Array.from({ length: MODEL_CONFIG.dModel }, (_, index) => 0.85 + index * 0.04);
}

export function tokenize(input: string): Token[] {
  const tokens: Token[] = [{ id: 0, text: "<BOS>", raw: "", unknown: false }];
  let index = 0;
  while (index < input.length && tokens.length < MODEL_CONFIG.maxTokens) {
    if (/\s/.test(input[index])) {
      index += 1;
      continue;
    }
    const match = sortedVocab.find((word) => input.startsWith(word, index));
    if (match) {
      tokens.push({ id: tokenToId.get(match)!, text: match, raw: match, unknown: false });
      index += match.length;
      continue;
    }
    const ascii = input.slice(index).match(/^[a-zA-Z0-9]+/);
    if (ascii) {
      tokens.push({ id: 2, text: ascii[0], raw: ascii[0], unknown: true });
      index += ascii[0].length;
      continue;
    }
    tokens.push({ id: 2, text: input[index], raw: input[index], unknown: true });
    index += 1;
  }
  return tokens;
}

function attentionForHead(
  normalized: Matrix,
  layer: number,
  head: number,
): { q: Matrix; k: Matrix; v: Matrix; rawScores: number[][]; scores: number[][]; weights: number[][]; context: Matrix } {
  const config = weights[layer];
  const q = multiply(normalized, config.wq[head]);
  const k = multiply(normalized, config.wk[head]);
  const v = multiply(normalized, config.wv[head]);
  const rawScores = q.map((query) => k.map((key) => {
    let dot = 0;
    for (let d = 0; d < HEAD_DIM; d += 1) dot += query[d] * key[d];
    return dot / Math.sqrt(HEAD_DIM);
  }));
  const scores = rawScores.map((row, i) => row.map((value, j) => j > i ? Number.NEGATIVE_INFINITY : value));
  const attentionWeights = scores.map(softmax);
  const context = multiply(attentionWeights, v);
  return { q, k, v, rawScores, scores, weights: attentionWeights, context };
}

export function forward(input: string | Token[]): ForwardTrace {
  const tokens = typeof input === "string" ? tokenize(input) : input;
  const tokenEmbeddings = tokens.map((token) => [...tokenEmbedding[token.id]]);
  const positionEmbeddings = tokens.map((_, index) => [...positionEmbedding[index]]);
  let hidden = add(tokenEmbeddings, positionEmbeddings);
  const embeddings = hidden;
  const layers: LayerTrace[] = [];

  for (let layer = 0; layer < MODEL_CONFIG.layers; layer += 1) {
    const inputHidden = hidden;
    const normalizedAttention = rmsNorm(inputHidden);
    const headResults = Array.from({ length: MODEL_CONFIG.heads }, (_, head) =>
      attentionForHead(normalizedAttention, layer, head),
    );
    const concatenated = headResults[0].context.map((row, i) => [
      ...row,
      ...headResults[1].context[i],
    ]);
    const projected = multiply(concatenated, weights[layer].wo);
    const afterAttention = add(inputHidden, projected);
    const normalizedMlp = rmsNorm(afterAttention);
    const preActivation = multiply(normalizedMlp, weights[layer].w1).map((row) =>
      row.map((value, j) => value + weights[layer].b1[j]),
    );
    const mlpHidden = preActivation.map((row) => row.map(gelu));
    const mlpOutput = multiply(mlpHidden, weights[layer].w2).map((row) =>
      row.map((value, j) => value + weights[layer].b2[j]),
    );
    hidden = add(afterAttention, mlpOutput);
    layers.push({
      input: inputHidden,
      normalizedAttention,
      queries: headResults.map((head) => head.q),
      keys: headResults.map((head) => head.k),
      values: headResults.map((head) => head.v),
      rawScores: headResults.map((head) => head.rawScores),
      scores: headResults.map((head) => head.scores),
      attention: headResults.map((head) => head.weights),
      contexts: headResults.map((head) => head.context),
      concatenated,
      projected,
      attentionOutput: afterAttention,
      afterAttention,
      normalizedMlp,
      mlpPreActivation: preActivation,
      mlpHidden,
      mlpOutput,
      output: hidden,
    });
  }

  const finalNormalized = rmsNorm(hidden);
  const last = finalNormalized[finalNormalized.length - 1];
  const logits = unembedding.map((row, tokenIndex) => {
    let value = outputBias[tokenIndex];
    for (let d = 0; d < MODEL_CONFIG.dModel; d += 1) value += row[d] * last[d];
    return value;
  });

  return { tokens, embeddings, tokenEmbeddings, positionEmbeddings, layers, finalNormalized, logits };
}

export function getCandidates(logits: number[], temperature: number, topK: number): Candidate[] {
  const safeTemperature = Math.max(0.05, temperature);
  const ranked = logits
    .map((logit, id) => ({ id, token: VOCAB[id], logit }))
    .sort((a, b) => b.logit - a.logit);
  const k = Math.max(1, Math.min(topK, ranked.length));
  const pool = ranked.slice(0, k);
  const probabilities = softmax(pool.map((item) => item.logit / safeTemperature));
  return pool.map((item, index) => ({ ...item, probability: probabilities[index] }));
}

export function chooseCandidate(candidates: Candidate[], sample: boolean, randomValue = 0.5): Candidate {
  if (!sample) return candidates[0];
  let cursor = randomValue;
  for (const candidate of candidates) {
    cursor -= candidate.probability;
    if (cursor <= 0) return candidate;
  }
  return candidates[candidates.length - 1];
}

export function tokensToText(tokens: Token[]) {
  return tokens
    .filter((token) => token.id !== 0)
    .map((token) => token.raw)
    .join(" ");
}

