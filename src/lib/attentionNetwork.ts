export type AttentionArchitecture = "mha" | "mqa" | "gqa" | "mla" | "dsa" | "gdn" | "kda";
export type AttentionSparsity = "dense" | "block" | "sliding";

export interface AttentionState {
  arch: AttentionArchitecture;
  Hq: number;
  Hkv: number;
  dh: number;
  dc: number;
  dr: number;
  sparsity: AttentionSparsity;
  topk: number;
  head: number;
  selected: string;
  dim: boolean;
  zoom: number | null;
}

export interface GraphNode {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  short: string;
  title: string;
  shape: string;
  index: number;
  formula: string;
  input: string;
  output: string;
  explain: string;
  kernel?: string;
  badge?: string;
  kind?: "cache" | "recon" | "index" | "gate" | "state";
  head?: number;
  shared: boolean;
  mini: boolean;
  cache: boolean;
}

export interface GraphEdge {
  a: string;
  b: string;
  head?: number;
  shared?: boolean;
  bypass?: boolean;
  tone?: "rope" | "index" | "gate" | "loop";
}

export interface GraphRegion {
  type: "lane" | "shared";
  x: number;
  y: number;
  w: number;
  h: number;
  head?: number;
}

export interface GraphLabel {
  x: number;
  y: number;
  text: string;
  cls: string;
  anchor?: string;
  head?: number;
}

export interface CustomPath {
  kind: "bus" | "residual" | "input";
  nodes: string[];
  d: string;
  tone?: "q" | "kv" | "v" | "rope";
  labelX?: number;
  labelY?: number;
  label?: string;
}

export interface AttentionGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  regions: GraphRegion[];
  labels: GraphLabel[];
  custom: CustomPath[];
  W: number;
  H: number;
  dm: number;
  group: (head: number) => number;
  laneW: number;
}

interface NodeExtra {
  formula?: string;
  input?: string;
  explain?: string;
  kernel?: string;
  badge?: string;
  kind?: GraphNode["kind"];
  head?: number;
  shared?: boolean;
  mini?: boolean;
  cache?: boolean;
}

export interface ArchitectureMeta {
  label: string;
  title: string;
  subtitle: string;
  position: string;
  idea: string;
  cacheFormula: string;
  models: string;
  note: string;
  kernel: string;
}

export const ATTENTION_ARCH_ORDER: AttentionArchitecture[] = ["mha", "mqa", "gqa", "mla", "dsa", "gdn", "kda"];

export const ATTENTION_ARCHITECTURES: Record<AttentionArchitecture, ArchitectureMeta> = {
  mha: {
    label: "MHA", title: "MULTI-HEAD ATTENTION",
    subtitle: "每个 Head 独立的 Q／K／V —— KV Cache 的基准形态",
    position: "架构层 · Softmax 注意力 · KV 全量独立",
    idea: "每个 Head 拥有自己的 K、V 投影。表达力最强，KV Cache 最大：2 · Hq · dhead / token / 层。",
    cacheFormula: "2 · Hq · dhead", models: "GPT-2 / GPT-3、LLaMA-1、BERT 系",
    note: "MQA、GQA、MLA 都可以看作对 MHA 的 KV 做“共享／压缩”的不同方式。",
    kernel: "FA2 / FA3 · 昇腾 FIA / FAS",
  },
  mqa: {
    label: "MQA", title: "MULTI-QUERY ATTENTION",
    subtitle: "全部 Head 共享一对 K／V —— 共享的极端（Hkv = 1）",
    position: "架构层 · Softmax 注意力 · KV 全共享",
    idea: "所有 Head 复用同一份 K、V。KV Cache 降到 2 · dhead / token / 层，解码访存最小；代价是表达能力损失，故后来由 GQA 折中。",
    cacheFormula: "2 · dhead", models: "PaLM、StarCoder、Falcon",
    note: "MQA 是 GQA 令 Hkv = 1 的谱系端点；Query 仍然每个 Head 独立。",
    kernel: "FA2 / FA3 · 昇腾 FIA / FAS",
  },
  gqa: {
    label: "GQA", title: "GROUPED-QUERY ATTENTION",
    subtitle: "组内共享 K／V，组间独立 —— 质量与缓存的折中",
    position: "架构层 · Softmax 注意力 · KV 分组共享",
    idea: "Hq 个 Query Head 分成 Hkv 组，组内共享一对 K、V。Query 从不共享。KV Cache = 2 · Hkv · dhead / token / 层。",
    cacheFormula: "2 · Hkv · dhead", models: "LLaMA-2/3、Mistral、Qwen2、Gemma",
    note: "Block Sparse 模式下，稀疏块选择可在组内共享（GBSA），摊薄选块开销；全部 KV 仍需保存。",
    kernel: "FA2 / FA3 · 昇腾 FIA / FAS · FAG（GQA 变体）",
  },
  mla: {
    label: "MLA", title: "MULTI-HEAD LATENT ATTENTION",
    subtitle: "KV 压成低秩潜向量 cKV ＋ 解耦 RoPE Key",
    position: "架构层 · Softmax 注意力 · KV 低秩压缩",
    idea: "把 KV 联合压缩成 dc 维潜向量 cKV，位置信息由独立的 dr 维解耦 RoPE Key 携带。持久缓存只有 cKV 与 kR；各 Head 的完整 K／V 在计算时临时还原。",
    cacheFormula: "dc + dr", models: "DeepSeek-V2 / V3",
    note: "临时还原的完整 K／V 用虚线框标注，不等于持久缓存。",
    kernel: "FlashMLA · vLLM MLA 后端",
  },
  dsa: {
    label: "MLA+DSA", title: "MLA + DEEPSEEK SPARSE ATTENTION",
    subtitle: "MLA 之上加 Lightning Indexer ＋ Top-k 选择",
    position: "架构层 · 稀疏注意力（叠加在 MLA 之上）",
    idea: "轻量 Lightning Indexer 为每个 query 给可见历史 token 打分，仅 Top-k 个位置进入主注意力。主注意力计算随 k 而非全部 n 增长；全量 cKV 与 kR 仍需保存，稀疏只省计算。",
    cacheFormula: "dc + dr（同 MLA）＋索引器开销", models: "DeepSeek-V3.2-Exp",
    note: "Top-k 与因果约束共同决定主注意力实际读取的位置；索引器不是新的持久 KV Cache。",
    kernel: "vLLM SFA · 昇腾 SMLA / MQSMLA",
  },
  gdn: {
    label: "GDN", title: "GATED DELTANET",
    subtitle: "线性注意力：定长状态矩阵 ＋ Delta Rule ＋ 标量遗忘门",
    position: "架构层 · 线性注意力 · 与 Softmax 家族并列的另一范式",
    idea: "不保存 KV，不做 QKᵀ 两两打分。每个 Head 维护定长状态矩阵 S（d×d）：每步先按标量门 α 遗忘，再用 Delta Rule“读旧值 → 算误差 → 写入”。解码内存与序列长度无关。",
    cacheFormula: "无 KV Cache；每 Head 状态 S 为定长 d×d", models: "Qwen3-Next（与全注意力层 3:1 混合）",
    note: "训练时使用 chunkwise 并行形式，推理逐步递归；无 Softmax、无 [n,n] 注意力矩阵。",
    kernel: "fla chunkwise_gdn · fused_recurrent",
  },
  kda: {
    label: "KDA", title: "KIMI DELTA ATTENTION",
    subtitle: "GDN 的细化：遗忘门从每头标量升级为逐通道向量",
    position: "架构层 · 线性注意力 · GDN 的逐通道门控扩展",
    idea: "结构与 GDN 同构，关键差异是遗忘门 α 从 Head 级标量变为逐通道向量，状态的每个特征维度独立决定遗忘快慢。",
    cacheFormula: "无 KV Cache；每 Head 状态 S 为定长 d×d", models: "Kimi Linear（KDA : MLA = 3:1 混合）",
    note: "谱系：DeltaNet ＋ 标量衰减 = GDN；DeltaNet ＋ 对角（逐通道）衰减 = KDA。",
    kernel: "fla chunkwise_kda · fused_recurrent",
  },
};

export const INITIAL_ATTENTION_STATE: AttentionState = {
  arch: "mha", Hq: 2, Hkv: 2, dh: 4, dc: 8, dr: 2,
  sparsity: "dense", topk: 3,
  head: 0, selected: "norm", dim: false, zoom: null,
};

export function isLatentAttention(arch: AttentionArchitecture) {
  return arch === "mla" || arch === "dsa";
}

export function isLinearAttention(arch: AttentionArchitecture) {
  return arch === "gdn" || arch === "kda";
}

export function attentionCacheEntries(arch: AttentionArchitecture, state: AttentionState): number | null {
  const normalized = normalizeAttentionState({ ...state, arch });
  if (isLinearAttention(arch)) return null;
  if (isLatentAttention(arch)) return normalized.dc + normalized.dr;
  if (arch === "mqa") return 2 * normalized.dh;
  return 2 * normalized.Hkv * normalized.dh;
}

export function normalizeAttentionState(state: AttentionState): AttentionState {
  const Hq = [1, 2, 4, 8].includes(state.Hq) ? state.Hq : 2;
  const next: AttentionState = {
    ...state,
    Hq,
    dh: [2, 4, 8].includes(state.dh) ? state.dh : 4,
    dc: [4, 8, 16].includes(state.dc) ? state.dc : 8,
    dr: [2, 4].includes(state.dr) ? state.dr : 2,
    topk: [2, 3].includes(state.topk) ? state.topk : 3,
    head: Math.max(0, Math.min(state.head, Hq - 1)),
  };
  if (next.arch === "mha") {
    next.Hkv = Hq;
  } else if (next.arch === "mqa") {
    next.Hkv = 1;
  } else if (next.arch === "gqa") {
    const options = Array.from({ length: Hq }, (_, index) => index + 1)
      .filter((value) => Hq % value === 0 && (Hq === 1 || value < Hq));
    if (!options.includes(next.Hkv)) next.Hkv = Hq === 1 ? 1 : Hq / 2;
  } else {
    next.Hkv = Hq;
  }
  if (!["mha", "mqa", "gqa"].includes(next.arch)) next.sparsity = "dense";
  return next;
}

function maskContent(state: AttentionState) {
  if (state.sparsity === "block") return {
    short: "Block Mask · BSA", title: "块级稀疏 + 因果约束",
    formula: "Ŝ[i,j] = 选中块且 j≤i ? S[i,j] : −∞",
    explain: `以 B×B 块为单位决定算与不算，同时保持因果约束。${state.arch === "gqa" ? "GQA 下块选择可在组内共享（GBSA）。" : ""}省计算不省缓存——全部 KV 仍在。`,
    kernel: "cuDNN BSA · FA block-sparse",
  };
  if (state.sparsity === "sliding") return {
    short: "Sliding Window", title: "滑窗 + 因果约束",
    formula: "Ŝ[i,j] = i−w < j≤i ? S[i,j] : −∞",
    explain: "每个 query 只看最近 w 个 token（Mistral SWA）。实际模型常与全注意力层交替堆叠，兼顾局部效率与全局信息。",
  };
  return {
    short: "Causal Mask", title: "加因果 Mask",
    formula: "Ŝ[i,j] = j≤i ? S[i,j] : −∞",
    explain: "遮蔽未来 token，保留因果自回归约束。Dense 模式下每个 query 看到全部历史。",
  };
}

export function buildAttentionGraph(sourceState: AttentionState): AttentionGraph {
  const state = normalizeAttentionState(sourceState);
  const { Hq, dh } = state, dm = Hq * dh;
  const latent = isLatentAttention(state.arch), linear = isLinearAttention(state.arch);
  const nodes: GraphNode[] = [], edges: GraphEdge[] = [], regions: GraphRegion[] = [];
  const labels: GraphLabel[] = [], custom: CustomPath[] = [];
  let serial = 0;
  const laneW = latent ? 250 : linear ? 280 : 240;
  const gap = latent ? 176 : linear ? 260 : state.arch === "gqa" || state.arch === "mqa" ? 128 : 56;
  const pad = latent || linear ? 160 : 64, contentW = Hq * laneW + (Hq - 1) * gap;
  const W = Math.max(760, contentW + pad * 2), start = (W - contentW) / 2;
  const centers = Array.from({ length: Hq }, (_, head) => start + laneW / 2 + head * (laneW + gap));
  const cx = W / 2;

  function add(id: string, x: number, y: number, w: number, h: number, short: string, title: string, shape: string, extra: NodeExtra = {}) {
    const node: GraphNode = {
      id, x, y, w, h, short, title, shape, index: serial++,
      formula: extra.formula || short, input: extra.input || "—", output: shape,
      explain: extra.explain || title, kernel: extra.kernel, badge: extra.badge, kind: extra.kind,
      head: extra.head, shared: extra.shared || false,
      mini: extra.mini || false, cache: extra.cache || false,
    };
    nodes.push(node);
    return node;
  }
  function edge(a: string, b: string, extra: Omit<GraphEdge, "a" | "b"> = {}) { edges.push({ a, b, ...extra }); }
  function label(x: number, y: number, text: string, cls = "small-label", anchor = "start") { labels.push({ x, y, text, cls, anchor }); }
  const group = (head: number) => state.arch === "mha" ? head : state.arch === "mqa" ? 0 : Math.floor(head / (Hq / state.Hkv));

  const commonY = 90;
  add("norm", cx, commonY, 236, 70, "RMSNorm", "注意力前归一化", `[n,${dm}] → [n,${dm}]`, {
    formula: "X̂ = RMSNorm(X)", input: `X [n,${dm}]`,
    explain: "Pre-norm 归一化。原始 X 不归一化地沿黄色主干保留到注意力残差。",
  });
  label(42, 52, `${linear ? "LINEAR ATTENTION" : "MULTI-HEAD ATTENTION"} / ${ATTENTION_ARCHITECTURES[state.arch].label}`, "phase-label");
  label(cx, 16, "X · Block 输入，同时保留至残差", "small-label", "middle");

  let concatY: number;
  if (state.arch === "mha") {
    const avY = 735; concatY = 875;
    const laneTop = 190, laneBottom = 815, mask = maskContent(state);
    centers.forEach((x, head) => {
      regions.push({ type: "lane", x: x - laneW / 2, y: laneTop, w: laneW, h: laneBottom - laneTop, head });
      labels.push({ x: x - laneW / 2 + 10, y: 204, text: `HEAD ${head} · 独立 K／V`, cls: "head-title", head });
      const miniW = (laneW - 32) / 3;
      ["Q", "K", "V"].forEach((kind, index) => {
        const id = kind.toLowerCase() + "-" + head;
        add(id, x - laneW / 2 + 16 + miniW / 2 + index * miniW, 250, miniW - 5, 72, "W" + kind, `计算 ${kind}_${head}`, `[n,${dh}]`, {
          head, mini: true, cache: kind !== "Q", kind: kind === "Q" ? undefined : "cache",
          formula: `${kind}_${head} = X̂ W${kind},${head}`,
          input: `X̂ [n,${dm}] · W [dmodel,${dh}]`,
          explain: `第 ${head} 个 Head 的独立 ${kind} 投影。${kind === "Q" ? "MHA 中 Q／K／V 全部独立。" : "这份按 Head 独立的 K／V 是持久缓存的基准形态。"}`,
          kernel: kind === "Q" ? undefined : "缓存节点：每头独立 K／V",
        });
        edge("norm", id, { head });
      });
      add("scores-" + head, x, 390, laneW - 24, 68, "QKᵀ / √d", "QK 点积与缩放", "[n,n]", {
        head, formula: `S${head} = Q${head} K${head}ᵀ / √${dh}`,
        input: `Q${head} [n,${dh}] · K${head} [n,${dh}]`,
        explain: "每个 Head 独立计算 token 之间的注意力分数；[n,n] 是 Softmax 注意力二次复杂度的来源。",
        kernel: "FA2／FA3 可将 scores→AV 融合，避免完整矩阵落显存",
      });
      edge("q-" + head, "scores-" + head, { head }); edge("k-" + head, "scores-" + head, { head });
      add("mask-" + head, x, 492, laneW - 24, 68, mask.short, mask.title, "[n,n]", {
        head, formula: mask.formula, input: `S${head} [n,n]`, explain: mask.explain, kernel: mask.kernel,
      });
      edge("scores-" + head, "mask-" + head, { head });
      add("softmax-" + head, x, 594, laneW - 24, 68, "Softmax", "每行注意力归一化", "[n,n]", {
        head, formula: `A${head} = softmax(Ŝ${head})`, input: `Ŝ${head} [n,n]`, explain: "每个 query 行独立归一化为注意力权重。",
      });
      edge("mask-" + head, "softmax-" + head, { head });
      add("av-" + head, x, avY, laneW - 24, 68, "AV", "注意力权重乘 V", `[n,${dh}]`, {
        head, formula: `C${head} = A${head} V${head}`, input: `A${head} [n,n] · V${head} [n,${dh}]`,
        explain: "V 沿旁路保留，与 Softmax 权重在这里汇合。",
      });
      edge("softmax-" + head, "av-" + head, { head }); edge("v-" + head, "av-" + head, { head, bypass: true }); edge("av-" + head, "concat", { head });
    });
    label(cx, concatY - 45, `C₀ … C${Hq - 1} · 按特征维拼接`, "small-label", "middle");
  } else if (state.arch === "gqa" || state.arch === "mqa") {
    const avY = 920; concatY = 1060;
    const laneTop = 190, laneBottom = 995, mask = maskContent(state);
    regions.push({ type: "shared", x: 28, y: 326, w: W - 56, h: 180 });
    label(42, 348, `SHARED KV GROUPS / Hkv = ${state.Hkv}${state.sparsity === "block" ? " · 块选择可组内共享" : ""}`, "cache-tag");
    centers.forEach((x, head) => {
      const currentGroup = group(head);
      regions.push({ type: "lane", x: x - laneW / 2, y: laneTop, w: laneW, h: laneBottom - laneTop, head });
      labels.push({ x: x - laneW / 2 + 10, y: 204, text: `HEAD ${head} → GROUP ${currentGroup}`, cls: "head-title", head });
      add("q-" + head, x, 250, laneW - 40, 70, `X̂W_Q,${head}`, "独立 Query 投影", `[n,${dh}]`, {
        head, formula: `Q${head} = X̂ W_Q,${head}`, input: `X̂ [n,${dm}] · W_Q,${head} [dmodel,${dh}]`,
        explain: `${ATTENTION_ARCHITECTURES[state.arch].label} 不共享 Query。Q${head} 仍由本 Head 独立计算。`,
      });
      edge("norm", "q-" + head, { head });
      add("scores-" + head, x, 570, laneW - 24, 68, `Q${head}K_G${currentGroup}ᵀ`, "Q 与共享 K 点积", "[n,n]", {
        head, shared: true, formula: `S${head} = Q${head} K_G${currentGroup}ᵀ / √${dh}`,
        input: `Q${head} [n,${dh}] · K_G${currentGroup} [n,${dh}]`, explain: `Q${head} 读取第 ${currentGroup} 组共享 Key，而不是自己的独立 Key。`,
        kernel: "FA2／FA3 可将 scores→AV 融合，避免完整矩阵落显存",
      });
      edge("q-" + head, "scores-" + head, { head }); edge("k-g" + currentGroup, "scores-" + head, { head, shared: true });
      add("mask-" + head, x, 680, laneW - 24, 68, mask.short, mask.title, "[n,n]", {
        head, formula: mask.formula, input: `S${head} [n,n]`, explain: mask.explain, kernel: mask.kernel,
      });
      edge("scores-" + head, "mask-" + head, { head });
      add("softmax-" + head, x, 790, laneW - 24, 68, "Softmax", "每行注意力归一化", "[n,n]", {
        head, formula: `A${head} = softmax(Ŝ${head})`, input: `Ŝ${head} [n,n]`, explain: "同组 Head 的 Q 不同，因此 Attention 权重仍可不同。",
      });
      edge("mask-" + head, "softmax-" + head, { head });
      add("av-" + head, x, avY, laneW - 24, 68, `A${head}V_G${currentGroup}`, "乘共享 Value", `[n,${dh}]`, {
        head, shared: true, formula: `C${head} = A${head} V_G${currentGroup}`,
        input: `A${head} [n,n] · V_G${currentGroup} [n,${dh}]`, explain: `Q${head} 使用第 ${currentGroup} 组共享 Value。`,
      });
      edge("softmax-" + head, "av-" + head, { head }); edge("v-g" + currentGroup, "av-" + head, { head, shared: true, bypass: true }); edge("av-" + head, "concat", { head });
    });
    for (let currentGroup = 0; currentGroup < state.Hkv; currentGroup++) {
      const members = centers.filter((_, head) => group(head) === currentGroup);
      const gx = members.reduce((sum, value) => sum + value, 0) / members.length;
      const gw = Math.min(176, Math.max(142, laneW - 12)), miniW = (gw - 28) / 2;
      add("k-g" + currentGroup, gx - miniW / 2 - 3, 410, miniW, 66, "K_G" + currentGroup, "共享 Key 投影", `[n,${dh}]`, {
        mini: true, shared: true, cache: true, kind: "cache", formula: `K_G${currentGroup} = X̂ W_K,G${currentGroup}`,
        input: `X̂ [n,${dm}]`, explain: `该 Key 由第 ${currentGroup} 组的全部 Query Head 共同读取；持久缓存只存这一份。`, kernel: "缓存节点：KV Cache 的真实形态",
      });
      add("v-g" + currentGroup, gx + miniW / 2 + 3, 410, miniW, 66, "V_G" + currentGroup, "共享 Value 投影", `[n,${dh}]`, {
        mini: true, shared: true, cache: true, kind: "cache", formula: `V_G${currentGroup} = X̂ W_V,G${currentGroup}`,
        input: `X̂ [n,${dm}]`, explain: `该 Value 由第 ${currentGroup} 组的所有 Query Head 共同读取。`, kernel: "缓存节点：KV Cache 的真实形态",
      });
      edge("norm", "k-g" + currentGroup, { shared: true }); edge("norm", "v-g" + currentGroup, { shared: true });
    }
    const kvNodes = nodes.filter((node) => node.id.startsWith("k-g") || node.id.startsWith("v-g"));
    const kvXs = kvNodes.map((node) => node.x), q0 = nodes.find((node) => node.id === "q-0")!;
    custom.push({ kind: "bus", tone: "q", nodes: ["norm", ...centers.map((_, head) => "q-" + head)], d: `M ${cx - 24} 125 V 160 M ${Math.min(...centers)} 160 H ${Math.max(...centers)}` });
    const kvTrunk = Hq === 1 ? `M ${q0.x + q0.w / 2 + 12} 125 V 314` : `M ${cx + 24} 125 V 314`;
    custom.push({ kind: "bus", tone: "kv", nodes: ["norm", ...kvNodes.map((node) => node.id)], d: kvTrunk + ` M ${Math.min(...kvXs)} 314 H ${Math.max(...kvXs)}` });
    for (let currentGroup = 0; currentGroup < state.Hkv; currentGroup++) {
      const memberHeads = centers.map((_, head) => head).filter((head) => group(head) === currentGroup);
      const members = memberHeads.map((head) => centers[head]);
      const k = nodes.find((node) => node.id === "k-g" + currentGroup)!, v = nodes.find((node) => node.id === "v-g" + currentGroup)!;
      const gx = (k.x + v.x) / 2;
      const routeSide = (x: number, sourceX: number) => x === gx ? (sourceX < x ? -1 : 1) : (x < gx ? 1 : -1);
      const kc = members.map((x) => x + routeSide(x, k.x) * (laneW / 2 + 24));
      const vc = members.map((x) => x + routeSide(x, v.x) * (laneW / 2 + 48));
      custom.push({ kind: "bus", tone: "kv", nodes: ["k-g" + currentGroup, ...memberHeads.map((head) => "scores-" + head)], d: `M ${k.x - 12} 443 V 462 M ${Math.min(k.x - 12, ...kc)} 462 H ${Math.max(k.x - 12, ...kc)} ` + kc.map((x) => `M ${x} 462 V 570`).join(" ") });
      custom.push({ kind: "bus", tone: "v", nodes: ["v-g" + currentGroup, ...memberHeads.map((head) => "av-" + head)], d: `M ${v.x + 12} 443 V 478 M ${Math.min(v.x + 12, ...vc)} 478 H ${Math.max(v.x + 12, ...vc)} ` + vc.map((x) => `M ${x} 478 V ${avY}`).join(" ") });
    }
  } else if (latent) {
    const dsa = state.arch === "dsa", offset = dsa ? 250 : 0;
    const avY = 1390 + offset; concatY = 1540 + offset;
    const laneTop = 470 + offset, laneBottom = 1460 + offset;
    const qY = 530 + offset, reconY = 700 + offset, ropeY = 850 + offset;
    const scoreY = 990 + offset, maskY = 1110 + offset, softmaxY = 1230 + offset;
    const qBusY = 435 + offset, dkvBusY = dsa ? 310 : 360, krBusY = dsa ? 325 : 395;
    regions.push({ type: "shared", x: cx - 302, y: 160, w: 604, h: 170 + offset });
    label(cx - 286, 184, dsa ? "SHARED LATENT + SPARSE SELECTOR" : "SHARED COMPRESSED CACHE", "cache-tag");
    add("dkv", cx - 145, 240, 250, 72, "X̂W_DKV", "KV 联合下投影 → cKV", `[n,${state.dc}]`, {
      shared: true, cache: true, kind: "cache", formula: "cKV = X̂ W_DKV", input: `X̂ [n,${dm}] · W_DKV [dmodel,${state.dc}]`,
      explain: "K／V 信息被联合压缩到低维 cKV；跨 token 持久缓存这份 latent，而不是每个 Head 的完整副本。", kernel: "持久缓存：cKV",
    });
    add("kr", cx + 145, 240, 250, 72, "RoPE(X̂W_KR)", "共享解耦 RoPE Key", `[n,${state.dr}]`, {
      shared: true, cache: true, kind: "cache", formula: "kR = RoPE(X̂ W_KR)", input: `X̂ [n,${dm}] · W_KR [dmodel,${state.dr}]`,
      explain: "解耦 RoPE Key 单独生成并缓存，每 token 持久缓存为 dc + dr 维。", kernel: "持久缓存：kR",
    });
    edge("norm", "dkv", { shared: true }); edge("norm", "kr", { shared: true, tone: "rope" });
    if (dsa) {
      add("indexer", cx, 380, 300, 72, "Lightning Indexer", "轻量历史位置打分", "I [n,n] · 低维／低精度", {
        shared: true, kind: "index", formula: "I[t,s] = q′ₜ · k′ₛ，s≤t", input: `X̂ [n,${dm}]（经独立低维投影）`,
        explain: "旁路索引器快速评估每个 query 与可见历史 token，决定主注意力“算哪些”；它不会替代全量 cKV 缓存。", kernel: "DSA 特有模块：Indexer 前向",
      });
      edge("norm", "indexer", { shared: true, tone: "index" });
      add("topk", cx, 500, 300, 72, `Top-k 选择 · k=${state.topk}`, "每个 query 选可见位置", "稀疏索引集 sel(t)", {
        shared: true, kind: "index", formula: `sel(t) = CausalTopK_s I[t,s]，|sel| ≤ ${state.topk}`, input: "I [n,n] + 因果约束",
        explain: "只把选中的位置交给主注意力。计算量随 k 而非全部 n 增长；全量 cKV 与 kR 仍需保存。", kernel: "vLLM SFA · 昇腾 SMLA 的稀疏选择部分",
      });
      edge("indexer", "topk", { shared: true, tone: "index" });
      label(cx, 340, "SPARSE SELECTOR · 决定“算哪些”", "small-label", "middle");
    }
    centers.forEach((x, head) => {
      regions.push({ type: "lane", x: x - laneW / 2, y: laneTop, w: laneW, h: laneBottom - laneTop, head });
      labels.push({ x: x - laneW / 2 + 10, y: laneTop + 22, text: `HEAD ${head} · 共享 cKV`, cls: "head-title", head });
      const miniW = (laneW - 36) / 2;
      add("qc-" + head, x - miniW / 2 - 3, qY, miniW, 68, "Q_C," + head, "内容 Query 投影", `[n,${dh}]`, {
        head, mini: true, formula: `Q_C,${head} = X̂ W_QC,${head}`, input: `X̂ [n,${dm}]`, explain: "当前 Head 的内容 Query 部分；教学图将 Query 的压缩—升维合并表示。",
      });
      add("qr-" + head, x + miniW / 2 + 3, qY, miniW, 68, "Q_R," + head, "RoPE Query 投影", `[n,${state.dr}]`, {
        head, mini: true, formula: `Q_R,${head} = RoPE(X̂ W_QR,${head})`, input: `X̂ [n,${dm}]`, explain: "当前 Head 的解耦 RoPE Query 部分，不进入 KV Cache。",
      });
      edge("norm", "qc-" + head, { head }); edge("norm", "qr-" + head, { head });
      add("kc-" + head, x - miniW / 2 - 3, reconY, miniW, 68, "cW_UK," + head, "从 cKV 还原内容 Key", `[n,${dh}]`, {
        head, mini: true, shared: true, kind: "recon", formula: `K_C,${head} = cKV W_UK,${head}`,
        input: `cKV [n,${state.dc}] · W_UK,${head} [dc,${dh}]`, explain: "从同一 cKV 用本 Head 的上游矩阵临时还原内容 Key；临时还原不占持久缓存。",
        kernel: "可通过矩阵吸收合并计算；图中展开是为了说明数据关系",
      });
      add("v-" + head, x + miniW / 2 + 3, reconY, miniW, 68, "cW_UV," + head, "从 cKV 还原 Value", `[n,${dh}]`, {
        head, mini: true, shared: true, kind: "recon", formula: `V${head} = cKV W_UV,${head}`,
        input: `cKV [n,${state.dc}] · W_UV,${head} [dc,${dh}]`, explain: "Value 也来自共享 latent，但上游投影按 Head 独立；临时还原 ≠ 持久缓存。",
      });
      edge("dkv", "kc-" + head, { head, shared: true }); edge("dkv", "v-" + head, { head, shared: true });
      add("rope-" + head, x, ropeY, laneW - 24, 68, "Concat + RoPE", "拼成完整 Q／K", `[n,${dh + state.dr}]`, {
        head, formula: `Q${head}=[Q_C;Q_R] · K${head}=[K_C;kR]`,
        input: `Q_C/K_C [n,${dh}] + Q_R/kR [n,${state.dr}]`, explain: "各 Head 的内容部分与共享 kR 拼接，形成完整注意力 Q／K。",
      });
      edge("qc-" + head, "rope-" + head, { head }); edge("qr-" + head, "rope-" + head, { head }); edge("kc-" + head, "rope-" + head, { head }); edge("kr", "rope-" + head, { head, shared: true, tone: "rope" });
      add("scores-" + head, x, scoreY, laneW - 24, 68, dsa ? "Selected QKᵀ" : "QKᵀ / √d", dsa ? "仅计算入选位置分数" : "完整 QK 点积", dsa ? "稀疏分数集合" : "[n,n]", {
        head, formula: dsa ? `S${head}[t,j] = Q${head}[t]K${head}[j]ᵀ / √${dh + state.dr}，j∈sel(t)` : `S${head} = Q${head} K${head}ᵀ / √${dh + state.dr}`,
        input: `Q,K [n,${dh + state.dr}]${dsa ? " + sel(t)" : ""}`,
        explain: dsa ? "主注意力只计算 Top-k 且满足因果约束的位置，而不是先计算完整稠密 QKᵀ 再丢弃。" : "在内容维与 RoPE 维拼接后的表示上计算分数。",
      });
      edge("rope-" + head, "scores-" + head, { head });
      add("mask-" + head, x, maskY, laneW - 24, 68, dsa ? `Sparse Mask · Top-${state.topk}` : "Causal Mask", dsa ? "Top-k 与因果联合约束" : "加因果 Mask", dsa ? "逻辑 [n,n] · 稀疏" : "[n,n]", {
        head, kind: dsa ? "index" : undefined,
        formula: dsa ? "Ŝ[t,j] = j∈sel(t) 且 j≤t ? S[t,j] : −∞" : "Ŝ = causal_mask(S)",
        input: dsa ? `稀疏分数 + sel(t)，k=${state.topk}` : `S${head} [n,n]`,
        explain: dsa ? "非选中位置与未来位置均为 −∞；Softmax 只在入选集合上归一化。" : "MLA 本身仍使用稠密因果 Mask。",
        kernel: dsa ? "vLLM SFA · 昇腾 SMLA / MQSMLA" : undefined,
      });
      edge("scores-" + head, "mask-" + head, { head });
      if (dsa) edge("topk", "mask-" + head, { head, shared: true, tone: "index" });
      add("softmax-" + head, x, softmaxY, laneW - 24, 68, "Softmax", dsa ? "仅在 Top-k 集合归一化" : "每行注意力归一化", dsa ? "稀疏权重" : "[n,n]", {
        head, formula: `A${head} = softmax(Ŝ${head})`, input: `Ŝ${head}`, explain: dsa ? "仅在 Top-k 个非 −∞ 位置上归一化。" : "每个 query 行独立归一化为注意力权重。",
      });
      edge("mask-" + head, "softmax-" + head, { head });
      add("av-" + head, x, avY, laneW - 24, 68, "AV", "注意力权重乘 V", `[n,${dh}]`, {
        head, formula: `C${head} = A${head} V${head}`, input: `A${head} · V${head} [n,${dh}]`, explain: "Value 来自本 Head 对共享 cKV 的临时还原。",
      });
      edge("softmax-" + head, "av-" + head, { head }); edge("v-" + head, "av-" + head, { head, bypass: true }); edge("av-" + head, "concat", { head });
    });
    label(cx, concatY - 45, `C₀ … C${Hq - 1} · 按特征维拼接`, "small-label", "middle");
    const dkvNode = nodes.find((node) => node.id === "dkv")!, krNode = nodes.find((node) => node.id === "kr")!;
    const qXs = nodes.filter((node) => node.id.startsWith("qc-") || node.id.startsWith("qr-")).map((node) => node.x);
    const kvChannels = centers.map((x) => x + (dkvNode.x < x ? -1 : 1) * (laneW / 2 + 36));
    const krChannels = centers.map((x) => x + (krNode.x < x ? -1 : 1) * (laneW / 2 + 72));
    custom.push({ kind: "bus", tone: "kv", nodes: ["dkv", ...nodes.filter((node) => node.id.startsWith("kc-") || node.id.startsWith("v-")).map((node) => node.id)], d: kvChannels.map((x) => latentChannelPath(dkvNode, x, dkvBusY, 590, reconY - 60, cx)).join(" ") });
    custom.push({ kind: "bus", tone: "rope", nodes: ["kr", ...centers.map((_, head) => "rope-" + head)], d: krChannels.map((x) => latentChannelPath(krNode, x, krBusY, 610, ropeY - 16, cx)).join(" ") });
    const qTrunkX = cx - 330;
    custom.push({ kind: "bus", tone: "q", nodes: ["norm", ...nodes.filter((node) => node.id.startsWith("qc-") || node.id.startsWith("qr-")).map((node) => node.id)], d: `M ${cx} 125 V 146 H ${qTrunkX} V ${qBusY} M ${Math.min(...qXs)} ${qBusY} H ${Math.max(...qXs)}` });
  } else {
    const kda = state.arch === "kda";
    const qkvY = 250, gateY = 370, state0Y = 520, readY = 625, deltaY = 745, writeY = 880, state1Y = 1020, outY = 1150;
    concatY = 1320;
    label(cx, 158, "无 QKᵀ [n,n] · 无 Softmax · 无 KV Cache · 每 Head 定长状态 S[d,d]", "cache-tag", "middle");
    centers.forEach((x, head) => {
      regions.push({ type: "lane", x: x - laneW / 2, y: 190, w: laneW, h: 1050, head });
      labels.push({ x: x - laneW / 2 + 10, y: 206, text: `HEAD ${head} · 独立定长状态`, cls: "head-title", head });
      const miniW = (laneW - 32) / 3;
      (["q", "k", "v"] as const).forEach((kind, index) => {
        const id = `${kind}-${head}`;
        add(id, x - laneW / 2 + 16 + miniW / 2 + index * miniW, qkvY, miniW - 12, 72, `${kind}_t`, `${kind === "q" ? "状态读出" : kind === "k" ? "状态寻址" : "新信息内容"}投影`, `[n,${dh}]`, {
          head, mini: true, formula: `${kind}_t = X̂ W_${kind.toUpperCase()},${head}`, input: `X̂ [n,${dm}]`,
          explain: kind === "q" ? "用 q 从状态矩阵读取输出，不参与 token 两两打分。" : kind === "k" ? "k 决定读写状态矩阵的“位置”。" : "v 是要写入状态的新信息。",
        });
        edge("norm", id, { head });
      });
      const gates = [
        { key: "a", short: kda ? "α_t ∈ ℝ^d" : "α_t ∈ ℝ", title: kda ? "逐通道遗忘门" : "每头标量遗忘门", shape: kda ? `[n,${dh}]` : "[n,1]", formula: kda ? "α_t = sigmoid(W_α x_t) ∈ ℝ^d" : "α_t = sigmoid(w_α x_t) ∈ ℝ", explain: kda ? "KDA 的关键差异：状态的每个特征通道独立决定遗忘快慢。" : "GDN 的整个状态矩阵按同一个标量速率遗忘。", kernel: "与 β 一起由小 MLP／线性层产生" },
        { key: "b", short: "β_t", title: "写入强度门", shape: "[n,1]", formula: "β_t = sigmoid(w_β x_t) ∈ (0,1)", explain: "决定本步误差信息以多大幅度写入状态。" },
        { key: "g", short: "g_t", title: "输出门", shape: `[n,${dh}]`, formula: "g_t = sigmoid(W_g x_t)", explain: "对从状态读出的输出逐通道调制。" },
      ];
      gates.forEach((gate, index) => {
        const id = `${gate.key}-${head}`;
        add(id, x - laneW / 2 + 16 + miniW / 2 + index * miniW, gateY, miniW - 12, 76, gate.short, gate.title, gate.shape, {
          head, mini: true, kind: "gate", badge: gate.key === "a" && kda ? "KDA" : undefined,
          formula: gate.formula, input: `x_t [${dm}]`, explain: gate.explain, kernel: gate.kernel,
        });
        edge("norm", id, { head, tone: "gate" });
      });
      add("state0-" + head, x, state0Y, laneW - 30, 68, "S_t₋₁", "上一步定长状态", `[${dh},${dh}]`, {
        head, kind: "state", badge: "STATE", formula: `S${head} ∈ ℝ^(${dh}×${dh})`, input: "上一 token 更新后的状态",
        explain: "不保存 KV 序列，只维护定长状态矩阵；无论序列多长，decode 状态大小都不变。", kernel: "推理：fused_recurrent 逐步更新",
      });
      add("read-" + head, x, readY, laneW - 30, 68, "v̂_t = Sᵀk_t", "Delta Rule · 读旧值", `[n,${dh}]`, {
        head, formula: "v̂_t = S_t₋₁ᵀ · k_t", input: `S_t₋₁ [${dh},${dh}] · k_t [${dh}]`, explain: "先用 k 从旧状态读出“当前已经记住的值”。",
      });
      edge("state0-" + head, "read-" + head, { head }); edge("k-" + head, "read-" + head, { head });
      add("delta-" + head, x, deltaY, laneW - 30, 68, "δ_t = v_t − v̂_t", "Delta Rule · 计算误差", `[n,${dh}]`, {
        head, formula: "δ_t = v_t − v̂_t", input: `v_t,v̂_t [${dh}]`, explain: "只把新信息中状态尚未记住的误差写入，这就是 Delta Rule 的名字来源。",
      });
      edge("read-" + head, "delta-" + head, { head }); edge("v-" + head, "delta-" + head, { head });
      add("write-" + head, x, writeY, laneW - 24, 78, kda ? "diag(α)S + βδkᵀ" : "αS + βδkᵀ", kda ? "逐通道遗忘 + 写入" : "标量遗忘 + 写入", `[${dh},${dh}]`, {
        head, kind: "state", badge: "STATE", formula: kda ? "S_t = diag(α_t) · S_t₋₁ + β_t · δ_t · k_tᵀ" : "S_t = α_t · S_t₋₁ + β_t · δ_t · k_tᵀ",
        input: "S_t₋₁ · α_t · β_t · δ_t · k_t",
        explain: `${kda ? "逐通道 α_t 让各特征独立遗忘。" : "标量 α_t 让整体统一遗忘。"}先衰减旧状态，再沿 k 方向写入 δ。`,
        kernel: `训练：fla chunkwise_${kda ? "kda" : "gdn"} 并行形式`,
      });
      edge("delta-" + head, "write-" + head, { head }); edge("state0-" + head, "write-" + head, { head, tone: "loop" });
      edge("k-" + head, "write-" + head, { head }); edge("a-" + head, "write-" + head, { head, tone: "gate" }); edge("b-" + head, "write-" + head, { head, tone: "gate" });
      add("state1-" + head, x, state1Y, laneW - 30, 68, "S_t", "更新后的定长状态", `[${dh},${dh}]`, {
        head, kind: "state", badge: "STATE", formula: "S_t → 下一 token 的 S_t₋₁", input: "write",
        explain: "递归环把状态送入下一 token；状态持续滚动更新，但尺寸始终不变，因此不需要随 n 增长的 KV Cache。", kernel: "定长递归状态，解码内存与 n 无关",
      });
      edge("write-" + head, "state1-" + head, { head }); edge("state1-" + head, "state0-" + head, { head, tone: "loop" });
      add("out-" + head, x, outY, laneW - 24, 72, "(q_tᵀS_t) ⊙ g_t", "状态读出 + 输出门", `[n,${dh}]`, {
        head, formula: "o_t = (q_tᵀ · S_t) ⊙ g_t", input: `q_t,g_t [${dh}] · S_t [${dh},${dh}]`, explain: "用 q 从更新后的状态读出本步输出，再经输出门 g 调制。",
      });
      edge("state1-" + head, "out-" + head, { head }); edge("q-" + head, "out-" + head, { head }); edge("g-" + head, "out-" + head, { head, tone: "gate" });
      edge("out-" + head, "concat", { head });
    });
    label(cx, concatY - 45, `o₀ … o${Hq - 1} · 拼接全部 Head`, "small-label", "middle");
  }

  add("concat", cx, concatY, 250, 70, "Concat", "拼接全部 Head", `[n,${dm}]`, {
    formula: `Z = [C₀ | … | C${Hq - 1}]`, input: `${Hq} × [n,${dh}]`, explain: `只沿特征维拼接，dmodel = Hq × dhead = ${dm}。`,
  });
  add("proj", cx, concatY + 100, 250, 70, "ZW_O", "注意力输出投影", `[n,${dm}]`, {
    formula: "P = Z W_O", input: `Z [n,${dm}] · W_O [dmodel,dmodel]`, explain: "混合所有 Head 的表示，得到注意力分支输出。",
  });
  edge("concat", "proj");
  add("residual", cx, concatY + 205, 250, 70, "X + P", "注意力残差相加", `[n,${dm}]`, {
    formula: "R = X + P", input: `X,P [n,${dm}]`, explain: "黄色 X 主干在此与注意力分支相加。",
  });
  edge("proj", "residual");
  custom.push({ kind: "residual", nodes: ["residual"], d: `M ${cx} 23 V 32 H 20 V ${concatY + 205} H ${cx - 125}`, labelX: 28, labelY: commonY - 49, label: "X · 主干旁路" });
  custom.push({ kind: "input", nodes: ["norm"], d: `M ${cx} 32 V 55` });
  label(42, concatY + 294, "MLP + RESIDUAL", "phase-label");
  add("mlpnorm", cx, concatY + 340, 250, 70, "RMSNorm", "MLP 前归一化", `[n,${dm}]`, {
    formula: "Ẑ = RMSNorm(R)", input: `R [n,${dm}]`, explain: "进入 MLP 前的第二次 Pre-norm。",
  });
  edge("residual", "mlpnorm");
  add("linear1", cx, concatY + 440, 250, 70, "ẐW₁ + b₁", "MLP 扩张", `[n,${dm * 2}]`, {
    formula: "U = Ẑ W₁ + b₁", input: `[n,${dm}] → [n,${dm * 2}]`, explain: "教学网络将 MLP 隐维设为 2 × dmodel。",
  });
  edge("mlpnorm", "linear1");
  add("gelu", cx, concatY + 540, 250, 70, "GELU", "逐元素非线性", `[n,${dm * 2}]`, {
    formula: "G = GELU(U)", input: `U [n,${dm * 2}]`, explain: "逐元素非线性变换。",
  });
  edge("linear1", "gelu");
  add("linear2", cx, concatY + 640, 250, 70, "GW₂ + b₂", "MLP 压缩", `[n,${dm}]`, {
    formula: "M = G W₂ + b₂", input: `[n,${dm * 2}] → [n,${dm}]`, explain: "将 MLP 表示压回 dmodel。",
  });
  edge("gelu", "linear2");
  add("mlpres", cx, concatY + 780, 250, 70, "(X + P) + M", "MLP 残差相加", `[n,${dm}]`, {
    formula: "X_next = (X + P) + M = R + M", input: "R = X + P；M = MLP(RMSNorm(R))",
    explain: "加入的是注意力残差的完整输出 R = X + P，不是原始 X；再与 MLP 输出 M 相加。",
  });
  edge("linear2", "mlpres");
  custom.push({ kind: "residual", nodes: ["residual", "mlpres"], d: `M ${cx - 36} ${concatY + 240} V ${concatY + 258} H 20 V ${concatY + 780} H ${cx - 125}`, labelX: 28, labelY: concatY + 218, label: "R = X + P · 旁路" });
  return { nodes, edges, regions, labels, custom, W, H: concatY + 870, dm, group, laneW };
}

function linearChannel(graph: AttentionGraph, node: GraphNode, side: -1 | 1, offset: number) {
  return { side, x: node.x + side * (graph.laneW / 2 + offset) };
}

function dsaSelectionPath(graph: AttentionGraph, a: GraphNode, b: GraphNode) {
  const side = b.x < graph.W / 2 ? -1 : 1;
  const channelX = b.x + side * (graph.laneW / 2 + 108);
  const busY = a.y + a.h / 2 + 22 + (b.head || 0) * 10;
  return `M ${a.x} ${a.y + a.h / 2} V ${busY} H ${channelX} V ${b.y} H ${b.x + side * b.w / 2}`;
}

function latentChannelPath(source: GraphNode, channelX: number, busY: number, bypassY: number, endY: number, centerX: number) {
  const detour = Math.abs(channelX - centerX) < 180;
  const waypointX = channelX < centerX ? centerX - 350 : centerX + 350;
  return `M ${source.x} ${source.y + source.h / 2} V ${busY} H ${detour ? waypointX : channelX} ${detour ? `V ${bypassY} H ${channelX}` : ""} V ${endY}`;
}

export function attentionEdgePath(graph: AttentionGraph, sourceState: AttentionState, edge: GraphEdge, a: GraphNode, b: GraphNode): string {
  const state = normalizeAttentionState(sourceState), latent = isLatentAttention(state.arch), linear = isLinearAttention(state.arch);
  const y1 = a.y + a.h / 2, y2 = b.y - b.h / 2, mid = Math.round((y1 + y2) / 2);
  const channel = (offset: number, sourceX = a.x) => {
    const center = latent && (b.id.startsWith("kc-") || b.id.startsWith("qc-")) ? b.x + b.w / 2 + 3
      : latent && (b.id.startsWith("v-") || b.id.startsWith("qr-")) ? b.x - b.w / 2 - 3 : b.x;
    const side = sourceX < center ? -1 : 1;
    return { side, x: center + side * (graph.laneW / 2 + offset) };
  };
  if (linear) {
    if (edge.a === "norm" && /^[abg]-/.test(b.id)) {
      const gateIndex = b.id.startsWith("a-") ? 0 : b.id.startsWith("b-") ? 1 : 2;
      const inputX = b.x - b.w / 2 - 6;
      const busY = 442 + gateIndex * 10;
      return `M ${a.x} ${y1} V 170 H ${inputX} V ${busY} H ${b.x} V ${b.y + b.h / 2}`;
    }
    if (a.id.startsWith("k-") && (b.id.startsWith("read-") || b.id.startsWith("write-"))) {
      const writing = b.id.startsWith("write-");
      const route = linearChannel(graph, b, 1, writing ? 48 : 24);
      const sourceX = a.x + (writing ? 12 : -12), busY = writing ? 318 : 306;
      const entryY = writing ? b.y : b.y - 14;
      return `M ${sourceX} ${y1} V ${busY} H ${route.x} V ${entryY} H ${b.x + b.w / 2}`;
    }
    if (a.id.startsWith("v-") && b.id.startsWith("delta-")) {
      const route = linearChannel(graph, b, 1, 72), entryY = b.y + 14;
      return `M ${a.x + a.w / 2} ${a.y} H ${route.x} V ${entryY} H ${b.x + b.w / 2}`;
    }
    if (a.id.startsWith("q-") && b.id.startsWith("out-")) {
      const route = linearChannel(graph, b, -1, 24), entryY = b.y - 18;
      return `M ${a.x - a.w / 2} ${a.y} H ${route.x} V ${entryY} H ${b.x - b.w / 2}`;
    }
    if (a.id.startsWith("a-") && b.id.startsWith("write-")) {
      const route = linearChannel(graph, b, -1, 48);
      return `M ${a.x - a.w / 2} ${a.y} H ${route.x} V ${b.y} H ${b.x - b.w / 2}`;
    }
    if (a.id.startsWith("b-") && b.id.startsWith("write-")) {
      const route = linearChannel(graph, b, -1, 120), entryY = b.y - 24;
      return `M ${a.x + 12} ${y1} V 426 H ${route.x} V ${entryY} H ${b.x - b.w / 2}`;
    }
    if (a.id.startsWith("g-") && b.id.startsWith("out-")) {
      const route = linearChannel(graph, b, 1, 96), entryY = b.y + 18;
      return `M ${a.x + a.w / 2} ${a.y} H ${route.x} V ${entryY} H ${b.x + b.w / 2}`;
    }
    if (edge.tone === "loop" && a.id.startsWith("state1-") && b.id.startsWith("state0-")) {
      const route = linearChannel(graph, a, -1, 96), sourceY = a.y + 12, entryY = b.y - 12;
      return `M ${a.x - a.w / 2} ${sourceY} H ${route.x} V ${entryY} H ${b.x - b.w / 2}`;
    }
    if (edge.tone === "loop" && a.id.startsWith("state0-") && b.id.startsWith("write-")) {
      const route = linearChannel(graph, a, -1, 72), sourceY = a.y + 12, entryY = b.y + 24;
      return `M ${a.x - a.w / 2} ${sourceY} H ${route.x} V ${entryY} H ${b.x - b.w / 2}`;
    }
    if (a.id.startsWith("state0-") && b.id.startsWith("read-")) return `M ${a.x - 24} ${y1} V ${y2}`;
    if (a.id.startsWith("state1-") && b.id.startsWith("out-")) return `M ${a.x - 24} ${y1} V ${y2}`;
  }
  if (state.arch === "mha" && b.id.startsWith("scores-") && (a.id.startsWith("q-") || a.id.startsWith("k-"))) return `M ${a.x} ${y1} V ${y2}`;
  if (state.arch === "gqa" || state.arch === "mqa") {
    if (edge.a === "norm") return `M ${b.x} ${edge.shared ? 314 : 160} V ${y2}`;
    if (b.id.startsWith("scores-") && a.id.startsWith("q-")) return `M ${a.x} ${y1} V ${y2}`;
    if (edge.shared && a.id.startsWith("k-g") && b.id.startsWith("scores-")) {
      const currentGroup = a.id.slice(3), v = graph.nodes.find((node) => node.id === "v-g" + currentGroup)!;
      const gx = (a.x + v.x) / 2, side = b.x === gx ? (a.x < b.x ? -1 : 1) : (b.x < gx ? 1 : -1);
      const channelX = b.x + side * (graph.laneW / 2 + 24);
      return `M ${channelX} ${b.y} H ${b.x + side * b.w / 2}`;
    }
    if (edge.bypass) {
      const currentGroup = a.id.slice(3), k = graph.nodes.find((node) => node.id === "k-g" + currentGroup)!;
      const gx = (a.x + k.x) / 2, side = b.x === gx ? (a.x < b.x ? -1 : 1) : (b.x < gx ? 1 : -1);
      const channelX = b.x + side * (graph.laneW / 2 + 48);
      return `M ${channelX} ${b.y} H ${b.x + side * b.w / 2}`;
    }
  }
  if (latent) {
    const dsa = state.arch === "dsa", offset = dsa ? 250 : 0;
    const qBusY = 435 + offset;
    if (edge.a === "norm" && (edge.b === "dkv" || edge.b === "kr")) {
      const sourceX = a.x + (edge.b === "dkv" ? -36 : 36);
      return `M ${sourceX} ${y1} V 146 H ${b.x} V ${y2}`;
    }
    if (edge.a === "norm" && (b.id.startsWith("qc-") || b.id.startsWith("qr-"))) return `M ${b.x} ${qBusY} V ${y2}`;
    if (edge.a === "dkv" && (b.id.startsWith("kc-") || b.id.startsWith("v-"))) {
      const route = channel(36);
      return `M ${route.x} ${b.y - 60} H ${b.x} V ${y2}`;
    }
    if (edge.a === "kr" && b.id.startsWith("rope-")) {
      const route = channel(72);
      return `M ${route.x} ${b.y - 16} H ${b.x + route.side * b.w / 2}`;
    }
    if ((a.id.startsWith("qc-") || a.id.startsWith("qr-")) && b.id.startsWith("rope-")) {
      const side = a.id.startsWith("qc-") ? -1 : 1, channelX = b.x + side * (graph.laneW / 2 + 16), entryY = b.y + (side < 0 ? -12 : 12);
      return `M ${a.x + side * a.w / 2} ${a.y} H ${channelX} V ${entryY} H ${b.x + side * b.w / 2}`;
    }
    if (a.id.startsWith("kc-") && b.id.startsWith("rope-")) return `M ${a.x} ${y1} V ${y2}`;
    if (edge.a === "topk" && b.id.startsWith("mask-")) return dsaSelectionPath(graph, a, b);
  }
  if (edge.bypass) {
    const side = latent ? (b.x < graph.W / 2 ? -1 : 1) : 1;
    const channelX = b.x + side * (graph.laneW / 2 + (latent ? 36 : 32));
    if (latent) return `M ${a.x} ${y1} V ${a.y + 70} H ${channelX} V ${b.y} H ${b.x + side * b.w / 2}`;
    return `M ${a.x + side * a.w / 2} ${a.y} H ${channelX} V ${b.y} H ${b.x + side * b.w / 2}`;
  }
  if (y2 >= y1) {
    const targetX = b.x;
    let pathY = mid;
    const horizontal = y1 === pathY ? targetX - a.x : 0;
    if (horizontal !== 0) {
      const left = Math.min(a.x, targetX), right = Math.max(a.x, targetX);
      const blocked = graph.nodes.some((node) => node.id !== a.id && node.id !== b.id && node.y - node.h / 2 - 5 <= pathY && pathY <= node.y + node.h / 2 + 5 && node.x - node.w / 2 - 5 < right && node.x + node.w / 2 + 5 > left);
      if (blocked) pathY = y2 - 12;
    }
    return `M ${a.x} ${y1} V ${pathY} H ${targetX} V ${y2}`;
  }
  const rightX = Math.min(graph.W - 12, Math.max(a.x + a.w / 2, b.x + b.w / 2) + 12 + (a.head || 0) * 2);
  return `M ${a.x + a.w / 2} ${a.y} H ${rightX} V ${b.y} H ${b.x + b.w / 2}`;
}

export function fullAttentionEdgePath(graph: AttentionGraph, sourceState: AttentionState, edge: GraphEdge, a: GraphNode, b: GraphNode): string {
  const state = normalizeAttentionState(sourceState), latent = isLatentAttention(state.arch);
  const direct = attentionEdgePath(graph, state, edge, a, b), y2 = b.y - b.h / 2;
  if (state.arch === "gqa" || state.arch === "mqa") {
    if (edge.a === "norm" && !edge.shared) return `M ${graph.W / 2 - 24} 125 V 160 H ${b.x} V ${y2}`;
    if (edge.a === "norm" && edge.shared) {
      const q0 = graph.nodes.find((node) => node.id === "q-0")!;
      const sourceX = state.Hq === 1 ? q0.x + q0.w / 2 + 12 : graph.W / 2 + 24;
      return `M ${sourceX} 125 V 314 H ${b.x} V ${y2}`;
    }
    if (a.id.startsWith("k-g") && b.id.startsWith("scores-")) {
      const v = graph.nodes.find((node) => node.id === "v-g" + a.id.slice(3))!, gx = (a.x + v.x) / 2;
      const side = b.x === gx ? (a.x < b.x ? -1 : 1) : (b.x < gx ? 1 : -1), channelX = b.x + side * (graph.laneW / 2 + 24);
      return `M ${a.x - 12} 443 V 462 H ${channelX} V ${b.y} H ${b.x + side * b.w / 2}`;
    }
    if (a.id.startsWith("v-g") && b.id.startsWith("av-")) {
      const k = graph.nodes.find((node) => node.id === "k-g" + a.id.slice(3))!, gx = (a.x + k.x) / 2;
      const side = b.x === gx ? (a.x < b.x ? -1 : 1) : (b.x < gx ? 1 : -1), channelX = b.x + side * (graph.laneW / 2 + 48);
      return `M ${a.x + 12} 443 V 478 H ${channelX} V ${b.y} H ${b.x + side * b.w / 2}`;
    }
  }
  if (latent) {
    const dsa = state.arch === "dsa", offset = dsa ? 250 : 0;
    const qBusY = 435 + offset, dkvBusY = dsa ? 310 : 360, krBusY = dsa ? 325 : 395;
    if (edge.a === "norm" && (b.id.startsWith("qc-") || b.id.startsWith("qr-"))) {
      const qTrunkX = graph.W / 2 - 330;
      return `M ${graph.W / 2} 125 V 146 H ${qTrunkX} V ${qBusY} H ${b.x} V ${y2}`;
    }
    if (edge.a === "dkv" && (b.id.startsWith("kc-") || b.id.startsWith("v-"))) {
      const center = b.id.startsWith("kc-") ? b.x + b.w / 2 + 3 : b.x - b.w / 2 - 3;
      const side = a.x < center ? -1 : 1, channelX = center + side * (graph.laneW / 2 + 36);
      return `${latentChannelPath(a, channelX, dkvBusY, 590, b.y - 60, graph.W / 2)} H ${b.x} V ${y2}`;
    }
    if (edge.a === "kr" && b.id.startsWith("rope-")) {
      const side = a.x < b.x ? -1 : 1, channelX = b.x + side * (graph.laneW / 2 + 72);
      return `${latentChannelPath(a, channelX, krBusY, 610, b.y - 16, graph.W / 2)} H ${b.x + side * b.w / 2}`;
    }
    if (edge.a === "topk" && b.id.startsWith("mask-")) return dsaSelectionPath(graph, a, b);
  }
  return direct;
}
