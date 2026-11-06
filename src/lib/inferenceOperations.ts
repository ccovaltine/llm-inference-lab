import { softmax as softmaxRow } from "@/lib/numeric";
import {
  MODEL_CONFIG, VOCAB, getEmbeddingParameters, getLayerParameters, getRmsGains,
  getUnembeddingParameters, type Candidate, type ForwardTrace, type Matrix,
} from "@/lib/tinyTransformer";

export const OPERATION_COUNT = 43;

export type OperationKind =
  | "tokenize" | "lookupToken" | "lookupPosition" | "add" | "rmsnorm"
  | "matmul" | "scores" | "mask" | "softmax" | "av" | "concat"
  | "gelu" | "lmhead" | "temperature" | "topk" | "sample" | "append";

export type TensorRole = "input" | "parameter" | "output";
export type OperationTensor = {
  name: string;
  data: Matrix;
  shape: string;
  role: TensorRole;
  rowLabels?: string[];
  colLabels?: string[];
  masked?: boolean;
};

export type Operation = {
  index: number;
  id: string;
  title: string;
  short: string;
  group: string;
  kind: OperationKind;
  layer?: number;
  headRelevant?: boolean;
  formula: string;
  code: string;
  explanation: string;
  inputs: OperationTensor[];
  parameters: OperationTensor[];
  outputs: OperationTensor[];
};

export type NumericExpansion = {
  coordinate: string;
  expression: string;
  terms: string[];
  result: string;
  note: string;
};

export const fmt = (value: number) => Number.isFinite(value) ? value.toFixed(3) : "−∞";
const dims = (count: number, prefix = "d") => Array.from({ length: count }, (_, index) => `${prefix}${index}`);
const tokenRows = (trace: ForwardTrace) => trace.tokens.map((token, index) => `${index} · ${token.text}`);
const shapeOf = (data: Matrix) => `[${data.length},${data[0]?.length ?? 0}]`;

function tensor(
  name: string,
  data: Matrix,
  role: TensorRole,
  options: Partial<Omit<OperationTensor, "name" | "data" | "role" | "shape">> = {},
): OperationTensor {
  return { name, data, role, shape: shapeOf(data), colLabels: dims(data[0]?.length ?? 0), ...options };
}

function seqTensor(name: string, data: Matrix, role: TensorRole, trace: ForwardTrace, options: Partial<OperationTensor> = {}) {
  return tensor(name, data, role, { rowLabels: tokenRows(trace), ...options });
}

function meta(index: number): Pick<Operation, "id" | "title" | "short" | "group" | "kind" | "formula" | "code" | "explanation"> {
  const common = (id: string, title: string, short: string, group: string, kind: OperationKind, formula: string, code: string, explanation: string) =>
    ({ id, title, short, group, kind, formula, code, explanation });
  if (index === 0) return common("tokenize", "分词与词表查表", "Tokenize", "输入与表示", "tokenize", "text → tokens → ids", `tokens = ["<BOS>", ...longestMatch(text)]\nids = tokens.map(token => vocab[token])`, "先按最长匹配切分文本，再把每个 token 换成词表中的整数 ID。未知文本保留其位置，并映射为 <UNK>。");
  if (index === 1) return common("token-embed", "查 Token Embedding", "Lookup Eₜ", "输入与表示", "lookupToken", "T[i,d] = E_token[ids[i],d]", `T = E_token[ids]  // gather rows\n// [n] → [n,8]，没有矩阵乘法`, "用每个 token ID 直接取出词嵌入矩阵的一整行。这一步是查表，不是 one-hot 与矩阵做乘法。");
  if (index === 2) return common("position-embed", "查 Position Embedding", "Lookup Eₚ", "输入与表示", "lookupPosition", "P[i,d] = E_position[i,d]", `P = E_position[0:n]  // gather by position\n// [n] → [n,8]`, "用位置下标 0…n−1 取出位置向量。相同 token 在不同位置会得到不同的位置向量。");
  if (index === 3) return common("embed-add", "词向量与位置向量相加", "T + P", "输入与表示", "add", "X[i,d] = T[i,d] + P[i,d]", `X = T + P  // element-wise\n// [n,8] + [n,8] → [n,8]`, "两份矩阵按相同位置、相同维度逐元素相加，形成进入第 1 层的主干表示 X。");
  if (index >= 36) {
    const outputMeta = [
      ["final-norm", "最终 RMSNorm", "RMSNorm", "rmsnorm", "H = RMSNorm(X₂)", `H = RMSNorm(X2)\n// [n,8] → [n,8]`, "对第 2 层输出的每个 token 独立归一化。之后只取最后一个位置计算下一个 token。"],
      ["logits", "LM Head：投影到词表", "h · W_Uᵀ + b", "lmhead", "logits[v] = H[n−1,:] · W_U[v,:] + b[v]", `h = H[n-1]             // [8]\nlogits = W_U @ h + b  // [29,8]@[8] → [29]`, "最后位置的 8 维向量分别与词表中 29 个 Unembedding 向量点积，再加上各 token 的 bias。"],
      ["temperature", "Temperature 缩放", "logits / T", "temperature", "z[v] = logits[v] / temperature", `z = logits / temperature\n// 温度只缩放分数，不改变候选排序`, "温度小于 1 会拉大选间差距，分布更尖锐；大于 1 会缩小差距，分布更平坦。"],
      ["topk", "Top-k 截断", "Keep k", "topk", "ẑ[v] = z[v]（v∈Top-k），否则 −∞", `keep = argsort(z)[-k:]\nz_hat = where(v in keep, z, -Infinity)`, "按缩放后的 logit 排名，只保留 k 个候选，其余位置写成 −∞，下一步 softmax 后概率即为 0。"],
      ["softmax", "对 Top-k 候选做 Softmax", "Softmax", "softmax", "p[v] = exp(ẑ[v]−max) / Σᵤexp(ẑ[u]−max)", `p = softmax(z_hat)\n// 减去最大值保证数值稳定\n// 全部 29 项概率之和为 1`, "先减去当前最大有限分数，再计算指数和归一化。被 Top-k 排除的项 exp(−∞)=0。"],
      ["sample", "从概率分布选择 1 个 token", "Categorical", "sample", "token_id ~ Categorical(p)", `r = random(0,1)\ntoken_id = first v where r <= cumsum(p)[v]\n// 也可选择 argmax 作为确定性解码`, "把 [0,1] 随机数落入累计概率区间，选中对应 token。页面也允许直接指定候选，便于观察后续变化。"],
      ["append", "追加 token，形成下一轮上下文", "Append", "append", "context′ = concat(context, selected_token)", `context.append(selected_token)\n// 下一轮最后一个位置改变，logits 随之改变`, "将选中的 token 接到原上下文末尾。若达到最大长度或生成 EOS 则停止，否则进入下一轮推理。"],
    ] as const;
    const item = outputMeta[index - 36];
    return common(item[0], item[1], item[2], "输出与自回归", item[3] as OperationKind, item[4], item[5], item[6]);
  }

  const layer = index < 20 ? 0 : 1;
  const local = index - (layer === 0 ? 4 : 20);
  const group = `Transformer Block ${layer + 1}`;
  const prefix = `L${layer + 1}`;
  const entries = [
    ["norm-attn", `${prefix} · 注意力前 RMSNorm`, "RMSNorm", "rmsnorm", "X̂ = RMSNorm(X)", `X_hat = RMSNorm(X)\n// 独立处理每个 token 的 8 维向量`, "先计算该行 8 个数值的均方根，再逐维除以 RMS 并乘 gain。主干 X 保留用于残差。"],
    ["q", `${prefix} · 计算 Query`, "X̂W_Q", "matmul", "Q = X̂W_Q", `Q = X_hat @ W_Q[head]\n// [n,8]@[8,4] → [n,4]`, "每个输出元素都是 X̂ 的一行与 W_Q 的一列做点积。Q 表示当前位置要查询的特征。"],
    ["k", `${prefix} · 计算 Key`, "X̂W_K", "matmul", "K = X̂W_K", `K = X_hat @ W_K[head]\n// [n,8]@[8,4] → [n,4]`, "K 与 Q 来自同一输入，但使用不同权重。后续 QK 点积决定位置之间的匹配程度。"],
    ["v", `${prefix} · 计算 Value`, "X̂W_V", "matmul", "V = X̂W_V", `V = X_hat @ W_V[head]\n// [n,8]@[8,4] → [n,4]`, "V 是注意力权重最终取回的内容。Q/K 决定取多少，V 决定取回什么。"],
    ["scores", `${prefix} · QK 点积与缩放`, "QKᵀ / 2", "scores", "S[i,j] = (Q[i,:]·K[j,:]) / √4", `S = Q @ K.T / sqrt(4)\n// [n,4]@[4,n] → [n,n]`, "每个位置 i 的 Q 与每个位置 j 的 K 做 4 维点积，再乘 1/√4=0.5。此时尚未 Mask。"],
    ["mask", `${prefix} · 加因果 Mask`, "Causal mask", "mask", "Ŝ[i,j] = S[i,j]（j≤i），否则 −∞", `S_hat = S + causal_mask\n// j > i 的位置加 -Infinity`, "上三角未来位置被改成 −∞，保证当前位置不能利用未来 token。可见位置的数值不变。"],
    ["softmax", `${prefix} · 每行 Softmax`, "Softmax", "softmax", "A[i,j] = exp(Ŝ[i,j]−maxᵢ) / Σᵤexp(Ŝ[i,u]−maxᵢ)", `A = softmax(S_hat, axis=-1)\n// 每个 query 行独立归一化`, "每个 query 行单独做 softmax。被 Mask 的未来位置概率严格为 0，可见位置概率之和为 1。"],
    ["av", `${prefix} · 注意力权重乘 V`, "AV", "av", "C[i,d] = Σⱼ A[i,j]V[j,d]", `C = A @ V\n// [n,n]@[n,4] → [n,4]`, "对 V 的每一列按注意力权重加权求和，得到当前 head 的 Context。"],
    ["concat", `${prefix} · 拼接两个 Head`, "Concat", "concat", "Z = [C₀ | C₁]", `Z = concat([C0, C1], axis=1)\n// [n,4] + [n,4] → [n,8]`, "只沿特征维拼接，不做算术运算。前 4 列来自 Head 0，后 4 列来自 Head 1。"],
    ["proj", `${prefix} · 注意力输出投影`, "ZW_O", "matmul", "P = ZW_O", `P = Z @ W_O\n// [n,8]@[8,8] → [n,8]`, "将两个 head 拼接后的 8 维表示重新混合，得到可写回主干的注意力分支输出。"],
    ["res-attn", `${prefix} · 注意力残差相加`, "X + P", "add", "R = X + P", `R = X + P\n// 保留原始主干并加入注意力分支`, "原始输入与注意力投影逐元素相加，形状保持 n×8。"],
    ["norm-mlp", `${prefix} · MLP 前 RMSNorm`, "RMSNorm", "rmsnorm", "Ẑ = RMSNorm(R)", `Z_hat = RMSNorm(R)\n// 进入 MLP 前再次归一化`, "对注意力残差输出做第二次 Pre-norm，之后进入逐 token 的 MLP。"],
    ["linear1", `${prefix} · MLP 扩张与加 Bias`, "ẐW₁+b₁", "matmul", "U = ẐW₁ + b₁", `U = Z_hat @ W1 + b1\n// [n,8]@[8,16] + [16] → [n,16]`, "第一层全连接把每个 token 从 8 维扩张到 16 维，并对每个输出维加对应 bias。"],
    ["gelu", `${prefix} · GELU 非线性`, "GELU", "gelu", "G = GELU(U)", `G = GELU(U)\n// 对每个元素独立计算，无跨维或跨 token 运算`, "GELU 使用 tanh 近似逐元素变换。负值被平滑压低，正值按大小保留。"],
    ["linear2", `${prefix} · MLP 压缩与加 Bias`, "GW₂+b₂", "matmul", "M = GW₂ + b₂", `M = G @ W2 + b2\n// [n,16]@[16,8] + [8] → [n,8]`, "第二层全连接将 16 维激活压回 8 维，并加输出 bias，准备与主干残差相加。"],
    ["res-mlp", `${prefix} · MLP 残差相加`, "R + M", "add", "X_next = R + M", `X_next = R + M\n// [n,8] + [n,8] → [n,8]`, "注意力后的主干与 MLP 输出逐元素相加，得到当前 Transformer Block 的最终输出。"],
  ] as const;
  const item = entries[local];
  return { ...common(`${prefix}-${item[0]}`, item[1], item[2], group, item[3] as OperationKind, item[4], item[5], item[6]), layer, headRelevant: local >= 1 && local <= 7 } as Pick<Operation, "id" | "title" | "short" | "group" | "kind" | "formula" | "code" | "explanation">;
}

export function buildOperation(
  index: number,
  trace: ForwardTrace,
  head: number,
  temperature: number,
  topK: number,
  selectedCandidate: Candidate,
): Operation {
  const base = meta(index);
  const n = trace.tokens.length;
  const empty = { inputs: [] as OperationTensor[], parameters: [] as OperationTensor[], outputs: [] as OperationTensor[] };
  const embeddings = getEmbeddingParameters();
  const vocabRows = VOCAB.map((_, id) => [id]);
  if (index === 0) return { ...base, index, ...empty,
    parameters: [tensor("词表 ID（行名为 token）", vocabRows, "parameter", { rowLabels: [...VOCAB], colLabels: ["id"] })],
    outputs: [tensor("Token IDs", [trace.tokens.map((token) => token.id)], "output", { rowLabels: ["sequence"], colLabels: trace.tokens.map((token) => token.text) })],
  };
  if (index === 1) return { ...base, index,
    inputs: [tensor("Token IDs", [trace.tokens.map((token) => token.id)], "input", { rowLabels: ["sequence"], colLabels: trace.tokens.map((token) => token.text) })],
    parameters: [tensor("E_token", embeddings.token, "parameter", { rowLabels: [...VOCAB] })],
    outputs: [seqTensor("T · Token vectors", trace.tokenEmbeddings, "output", trace)],
  };
  if (index === 2) return { ...base, index,
    inputs: [tensor("Position IDs", [Array.from({ length: n }, (_, i) => i)], "input", { rowLabels: ["sequence"], colLabels: trace.tokens.map((token) => token.text) })],
    parameters: [tensor("E_position", embeddings.position, "parameter", { rowLabels: dims(MODEL_CONFIG.maxTokens, "pos") })],
    outputs: [seqTensor("P · Position vectors", trace.positionEmbeddings, "output", trace)],
  };
  if (index === 3) return { ...base, index,
    inputs: [seqTensor("T", trace.tokenEmbeddings, "input", trace), seqTensor("P", trace.positionEmbeddings, "input", trace)],
    parameters: [], outputs: [seqTensor("X", trace.embeddings, "output", trace)],
  };

  if (index >= 36) {
    const last = trace.finalNormalized.at(-1)!;
    const outputParams = getUnembeddingParameters();
    const scaled = trace.logits.map((value) => value / Math.max(0.05, temperature));
    const ranked = scaled.map((value, id) => ({ value, id })).sort((a, b) => b.value - a.value);
    const keep = new Set(ranked.slice(0, Math.max(1, Math.min(topK, VOCAB.length))).map((item) => item.id));
    const clipped = scaled.map((value, id) => keep.has(id) ? value : Number.NEGATIVE_INFINITY);
    const probabilities = softmaxRow(clipped);
    const vectorOutput = (name: string, values: number[]) => tensor(name, [values], "output", { rowLabels: ["vocab"], colLabels: [...VOCAB] });
    if (index === 36) return { ...base, index,
      inputs: [seqTensor("X₂", trace.layers[1].output, "input", trace)],
      parameters: [tensor("RMS gain", [getRmsGains()], "parameter", { rowLabels: ["gain"] }), tensor("ε", [[1e-5]], "parameter", { rowLabels: ["epsilon"], colLabels: ["value"] })],
      outputs: [seqTensor("H", trace.finalNormalized, "output", trace)],
    };
    if (index === 37) return { ...base, index,
      inputs: [tensor("h_last", [last], "input", { rowLabels: [`position ${n - 1}`] })],
      parameters: [tensor("W_U", outputParams.weights, "parameter", { rowLabels: [...VOCAB] }), tensor("b_U", [outputParams.bias], "parameter", { rowLabels: ["bias"], colLabels: [...VOCAB] })],
      outputs: [vectorOutput("Logits", trace.logits)],
    };
    if (index === 38) return { ...base, index,
      inputs: [tensor("Logits", [trace.logits], "input", { rowLabels: ["vocab"], colLabels: [...VOCAB] })],
      parameters: [tensor("temperature", [[temperature]], "parameter", { rowLabels: ["T"], colLabels: ["value"] })],
      outputs: [vectorOutput("Scaled logits z", scaled)],
    };
    if (index === 39) return { ...base, index,
      inputs: [tensor("Scaled logits z", [scaled], "input", { rowLabels: ["vocab"], colLabels: [...VOCAB] })],
      parameters: [tensor("k", [[topK]], "parameter", { rowLabels: ["top-k"], colLabels: ["value"] })],
      outputs: [vectorOutput("Clipped logits ẑ", clipped)],
    };
    if (index === 40) return { ...base, index,
      inputs: [tensor("Clipped logits ẑ", [clipped], "input", { rowLabels: ["vocab"], colLabels: [...VOCAB] })],
      parameters: [], outputs: [vectorOutput("概率 p（完整词表）", probabilities)],
    };
    if (index === 41) return { ...base, index,
      inputs: [tensor("概率 p", [probabilities], "input", { rowLabels: ["vocab"], colLabels: [...VOCAB] })],
      parameters: [], outputs: [tensor("Selected token ID", [[selectedCandidate.id]], "output", { rowLabels: [selectedCandidate.token], colLabels: ["id"] })],
    };
    return { ...base, index,
      inputs: [tensor("当前 Token IDs", [trace.tokens.map((token) => token.id)], "input", { rowLabels: ["context"], colLabels: trace.tokens.map((token) => token.text) }), tensor("选中 Token ID", [[selectedCandidate.id]], "input", { rowLabels: [selectedCandidate.token], colLabels: ["id"] })],
      parameters: [], outputs: [tensor("下一轮 Token IDs", [[...trace.tokens.map((token) => token.id), selectedCandidate.id]], "output", { rowLabels: ["context′"], colLabels: [...trace.tokens.map((token) => token.text), selectedCandidate.token] })],
    };
  }

  const layer = index < 20 ? 0 : 1;
  const local = index - (layer === 0 ? 4 : 20);
  const state = trace.layers[layer];
  const params = getLayerParameters(layer);
  const gains = [tensor("RMS gain", [getRmsGains()], "parameter", { rowLabels: ["gain"] }), tensor("ε", [[1e-5]], "parameter", { rowLabels: ["epsilon"], colLabels: ["value"] })];
  const seq = (name: string, data: Matrix, role: TensorRole, masked = false) => seqTensor(name, data, role, trace, { masked, colLabels: masked ? tokenRows(trace) : dims(data[0].length) });
  const linear = (inputName: string, input: Matrix, weightName: string, weight: Matrix, outputName: string, output: Matrix, bias?: number[]) => ({
    inputs: [seq(inputName, input, "input")],
    parameters: [tensor(weightName, weight, "parameter", { rowLabels: dims(weight.length, "k") }), ...(bias ? [tensor("bias", [bias], "parameter", { rowLabels: ["b"] })] : [])],
    outputs: [seq(outputName, output, "output")],
  });
  if (local === 0) return { ...base, index, inputs: [seq("X", state.input, "input")], parameters: gains, outputs: [seq("X̂", state.normalizedAttention, "output")] };
  if (local === 1) return { ...base, index, ...linear("X̂", state.normalizedAttention, `W_Q · head ${head}`, params.wq[head], `Q · head ${head}`, state.queries[head]) };
  if (local === 2) return { ...base, index, ...linear("X̂", state.normalizedAttention, `W_K · head ${head}`, params.wk[head], `K · head ${head}`, state.keys[head]) };
  if (local === 3) return { ...base, index, ...linear("X̂", state.normalizedAttention, `W_V · head ${head}`, params.wv[head], `V · head ${head}`, state.values[head]) };
  if (local === 4) return { ...base, index,
    inputs: [seq(`Q · head ${head}`, state.queries[head], "input"), seq(`K · head ${head}`, state.keys[head], "input")],
    parameters: [tensor("缩放系数 1/√4", [[0.5]], "parameter", { rowLabels: ["scale"], colLabels: ["value"] })],
    outputs: [seq("S · 未 Mask 原始分数", state.rawScores[head], "output", true)],
  };
  if (local === 5) {
    const causal = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => j > i ? Number.NEGATIVE_INFINITY : 0));
    return { ...base, index,
      inputs: [seq("S · 原始分数", state.rawScores[head], "input", true)],
      parameters: [seq("Causal mask（加法参数）", causal, "parameter", true)],
      outputs: [seq("Ŝ · Mask 后分数", state.scores[head], "output", true)],
    };
  }
  if (local === 6) return { ...base, index, inputs: [seq("Ŝ", state.scores[head], "input", true)], parameters: [], outputs: [seq("A · 注意力权重", state.attention[head], "output", true)] };
  if (local === 7) return { ...base, index, inputs: [seq("A", state.attention[head], "input", true), seq("V", state.values[head], "input")], parameters: [], outputs: [seq(`C · head ${head}`, state.contexts[head], "output")] };
  if (local === 8) return { ...base, index, inputs: [seq("C₀", state.contexts[0], "input"), seq("C₁", state.contexts[1], "input")], parameters: [], outputs: [seq("Z = [C₀|C₁]", state.concatenated, "output")] };
  if (local === 9) return { ...base, index, ...linear("Z", state.concatenated, "W_O", params.wo, "P", state.projected) };
  if (local === 10) return { ...base, index, inputs: [seq("X", state.input, "input"), seq("P", state.projected, "input")], parameters: [], outputs: [seq("R", state.afterAttention, "output")] };
  if (local === 11) return { ...base, index, inputs: [seq("R", state.afterAttention, "input")], parameters: gains, outputs: [seq("Ẑ", state.normalizedMlp, "output")] };
  if (local === 12) return { ...base, index, ...linear("Ẑ", state.normalizedMlp, "W₁", params.w1, "U · GELU 前", state.mlpPreActivation, params.b1) };
  if (local === 13) return { ...base, index, inputs: [seq("U", state.mlpPreActivation, "input")], parameters: [], outputs: [seq("G · GELU 后", state.mlpHidden, "output")] };
  if (local === 14) return { ...base, index, ...linear("G", state.mlpHidden, "W₂", params.w2, "M", state.mlpOutput, params.b2) };
  return { ...base, index, inputs: [seq("R", state.afterAttention, "input"), seq("M", state.mlpOutput, "input")], parameters: [], outputs: [seq(`X_${layer + 1}`, state.output, "output")] };
}

const productTerms = (a: number[], b: number[]) => a.map((value, index) => `(${fmt(value)} × ${fmt(b[index])})`);

export function buildExpansion(
  operation: Operation,
  row: number,
  col: number,
  trace: ForwardTrace,
): NumericExpansion {
  const output = operation.outputs[0];
  const r = Math.max(0, Math.min(row, output.data.length - 1));
  const c = Math.max(0, Math.min(col, output.data[0].length - 1));
  const value = output.data[r][c];
  const coordinate = `${output.name}[${r},${c}]`;
  const done = (expression: string, terms: string[], note: string): NumericExpansion => ({ coordinate, expression, terms, result: fmt(value), note });
  const input = operation.inputs[0];
  if (operation.kind === "matmul") {
    const a = input.data[r];
    const weight = operation.parameters[0].data.map((line) => line[c]);
    const terms = productTerms(a, weight);
    const bias = operation.parameters[1]?.data[0][c];
    return done(`Σₖ(${input.name}[${r},k] × ${operation.parameters[0].name}[k,${c}])${bias !== undefined ? ` + b[${c}]` : ""}`, [...terms, ...(bias !== undefined ? [`+ bias ${fmt(bias)}`] : [])], `共 ${a.length} 次乘法、${a.length - 1} 次加法${bias !== undefined ? "，最后再加 1 个 bias" : ""}。右侧参数矩阵中本次使用第 ${c} 列。`);
  }
  if (operation.kind === "scores") {
    const q = operation.inputs[0].data[r];
    const k = operation.inputs[1].data[c];
    return done(`(Q[${r},:] · K[${c},:]) × 0.5`, [...productTerms(q, k), "点积结果 × 0.500"], "Q 的当前行与 K 的当前列做 4 维点积，再统一缩放。这里尚未应用因果 Mask。 ");
  }
  if (operation.kind === "av") {
    const a = operation.inputs[0].data[r];
    const v = operation.inputs[1].data.map((line) => line[c]);
    return done(`Σⱼ A[${r},j] × V[j,${c}]`, productTerms(a, v), "对全部 key 位置加权求和；被 Mask 的位置因 A=0，乘积也为 0。 ");
  }
  if (operation.kind === "add") return done(`${fmt(operation.inputs[0].data[r][c])} + ${fmt(operation.inputs[1].data[r][c])}`, [], "两份输入张量按完全相同的位置逐元素相加，没有可训练参数。 ");
  if (operation.kind === "rmsnorm") {
    const source = input.data[r];
    const rms = Math.sqrt(source.reduce((sum, entry) => sum + entry * entry, 0) / source.length + 1e-5);
    const gain = getRmsGains()[c];
    return done(`${fmt(source[c])} / √((Σx²)/8 + 0.00001) × ${fmt(gain)}`, [...source.map((entry, index) => `x${index}² = ${fmt(entry * entry)}`), `Σx²/8 + ε = ${fmt(rms * rms)}`, `RMS = ${fmt(rms)}`], "分母由当前 token 的整行 8 个输入值共同计算，再对当前维度乘对应 gain。 ");
  }
  if (operation.kind === "lookupToken" || operation.kind === "lookupPosition") {
    const sourceRow = operation.kind === "lookupToken" ? trace.tokens[r].id : r;
    return done(`${operation.parameters[0].name}[${sourceRow},${c}] = ${fmt(operation.parameters[0].data[sourceRow][c])}`, [`输入索引 = ${sourceRow}`, `输出位置 = ${r}`, `输出维度 = ${c}`], "查表只复制参数矩阵中的一行，不做乘法或加法。 ");
  }
  if (operation.kind === "mask") {
    const raw = input.data[r][c];
    const masked = c > r;
    return done(masked ? `${fmt(raw)} + (−∞) = −∞` : `${fmt(raw)} + 0 = ${fmt(raw)}`, [`i = ${r}, j = ${c}`, masked ? "j > i：未来位置" : "j ≤ i：可见位置"], masked ? "Softmax 中 exp(−∞) = 0，因此该位置不会贡献信息。" : "可见位置的原始分数保持不变。 ");
  }
  if (operation.kind === "softmax") {
    const scores = input.data[r];
    const max = Math.max(...scores.filter(Number.isFinite));
    const exps = scores.map((entry) => Number.isFinite(entry) ? Math.exp(entry - max) : 0);
    const denominator = exps.reduce((sum, entry) => sum + entry, 0);
    return done(`exp(${fmt(scores[c])} − ${fmt(max)}) / ${fmt(denominator)}`, scores.map((entry, index) => `j${index}: exp(${fmt(entry)} − max) = ${fmt(exps[index])}`), `分母是当前 query 行全部 ${scores.length} 个指数项之和；本行输出概率总和为 1。`);
  }
  if (operation.kind === "concat") {
    const sourceHead = c < 4 ? 0 : 1;
    return done(`C${sourceHead}[${r},${c % 4}] → Z[${r},${c}]`, [`来源 Head = ${sourceHead}`, `来源列 = ${c % 4}`, `目标列 = ${c}`], "拼接仅复制数值并改变列的位置，没有参数，也不改变数值。 ");
  }
  if (operation.kind === "gelu") {
    const x = input.data[r][c];
    const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3);
    return done(`0.5 × ${fmt(x)} × (1 + tanh(${fmt(inner)}))`, [`x³ = ${fmt(x ** 3)}`, `0.044715x³ = ${fmt(0.044715 * x ** 3)}`, `tanh(${fmt(inner)}) = ${fmt(Math.tanh(inner))}`], "GELU 对当前元素独立计算，没有可训练参数。 ");
  }
  if (operation.kind === "lmhead") {
    const h = input.data[0];
    const w = operation.parameters[0].data[c];
    const bias = operation.parameters[1].data[0][c];
    return done(`W_U[${c},:] · h_last + b[${c}]`, [...productTerms(w, h), `+ bias ${fmt(bias)}`], `当前输出列对应候选 token「${VOCAB[c]}」。使用 W_U 的第 ${c} 行，而不是第 ${c} 列。`);
  }
  if (operation.kind === "temperature") return done(`${fmt(input.data[r][c])} / ${fmt(operation.parameters[0].data[0][0])}`, [], "每个 logit 都除以同一个温度参数；排序不变，但后续概率分布会改变。 ");
  if (operation.kind === "topk") {
    const scores: number[] = input.data[0];
    const k = operation.parameters[0].data[0][0];
    const sorted = [...scores].sort((a, b) => b - a);
    const threshold = sorted[k - 1];
    return done(Number.isFinite(value) ? `${fmt(scores[c])} ≥ 第 k 名阈值 ${fmt(threshold)} → 保留` : `${fmt(scores[c])} 不在 Top-${k} → 写为 −∞`, [`k = ${k}`, `第 k 名分数 = ${fmt(threshold)}`, `候选 token = ${VOCAB[c]}`], "截断后的 −∞ 会在下一步 Softmax 中转化为 0 概率。 ");
  }
  if (operation.kind === "sample") return done(`selected_token = ${VOCAB[output.data[r][c]]} (ID ${output.data[r][c]})`, [`累计概率区间按词表顺序累加`, `当前候选由页面选中或随机采样决定`], "采样只输出一个整数 ID，不再产生隐藏向量。 ");
  if (operation.kind === "append") return done(`context[${c}] = ${output.data[r][c]}`, [`新序列长度 = ${output.data[0].length}`], c === output.data[0].length - 1 ? "最后一项是本轮选中的新 token。" : "已有 token 原样保留，顺序不变。 ");
  return done(`ID = ${output.data[r][c]}`, [`对应 token = ${VOCAB[output.data[r][c]] ?? trace.tokens[c]?.text}`], "词表查表将文本映射为整数 ID。 ");
}
