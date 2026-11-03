export type AttentionArchitecture = "mha" | "gqa" | "mla";

export interface AttentionState {
  arch: AttentionArchitecture;
  Hq: number;
  Hkv: number;
  dh: number;
  dc: number;
  dr: number;
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
  tone?: "rope";
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
  head?: number;
  shared?: boolean;
  mini?: boolean;
  cache?: boolean;
}

export const INITIAL_ATTENTION_STATE: AttentionState = {
  arch: "mha", Hq: 2, Hkv: 2, dh: 4, dc: 8, dr: 2,
  head: 0, selected: "norm", dim: false, zoom: null,
};

export function normalizeAttentionState(state: AttentionState): AttentionState {
  const next = { ...state, head: Math.min(state.head, state.Hq - 1) };
  if (next.arch === "mha") {
    next.Hkv = next.Hq;
  } else {
    const options = Array.from({ length: next.Hq }, (_, index) => index + 1)
      .filter((value) => next.Hq % value === 0);
    if (!options.includes(next.Hkv) || (next.Hkv === next.Hq && next.Hq > 1)) {
      next.Hkv = next.Hq === 1 ? 1 : next.Hq / 2;
    }
  }
  return next;
}

export function buildAttentionGraph(state: AttentionState): AttentionGraph {
  const Hq = state.Hq, dh = state.dh, dm = Hq * dh;
  const nodes: GraphNode[] = [], edges: GraphEdge[] = [], regions: GraphRegion[] = [];
  const labels: GraphLabel[] = [], custom: CustomPath[] = [];
  let serial = 0;
  const laneW = state.arch === "mla" ? 250 : 240;
  const gap = state.arch === "mla" ? 176 : state.arch === "gqa" ? 128 : 56;
  const pad = 64, contentW = Hq * laneW + (Hq - 1) * gap;
  const W = Math.max(760, contentW + pad * 2), start = (W - contentW) / 2;
  const centers = Array.from({ length: Hq }, (_, head) => start + laneW / 2 + head * (laneW + gap));
  const cx = W / 2;

  function add(id: string, x: number, y: number, w: number, h: number, short: string, title: string, shape: string, extra: NodeExtra = {}) {
    const node: GraphNode = {
      id, x, y, w, h, short, title, shape, index: serial++,
      formula: extra.formula || short, input: extra.input || "—", output: shape,
      explain: extra.explain || title, head: extra.head, shared: extra.shared || false,
      mini: extra.mini || false, cache: extra.cache || false,
    };
    nodes.push(node);
    return node;
  }
  function edge(a: string, b: string, extra: Omit<GraphEdge, "a" | "b"> = {}) { edges.push({ a, b, ...extra }); }
  function label(x: number, y: number, text: string, cls = "small-label", anchor = "start") { labels.push({ x, y, text, cls, anchor }); }
  const group = (head: number) => Math.floor(head / (Hq / state.Hkv));

  const commonY = 90;
  add("norm", cx, commonY, 236, 70, "RMSNorm", "注意力前归一化", `[n,${dm}] → [n,${dm}]`, {
    formula: "X̂ = RMSNorm(X)", input: `X [n,${dm}]`,
    explain: "Pre-norm 归一化。原始 X 不归一化地沿黄色主干保留到注意力残差。",
  });
  label(42, 52, "MULTI-HEAD ATTENTION / " + state.arch.toUpperCase(), "phase-label");
  label(cx, 16, "X · Block 输入，同时保留至残差", "small-label", "middle");

  let avY: number, concatY: number;
  if (state.arch === "mha") {
    avY = 735; concatY = 875;
    const laneTop = 190, laneBottom = 815;
    centers.forEach((x, head) => {
      regions.push({ type: "lane", x: x - laneW / 2, y: laneTop, w: laneW, h: laneBottom - laneTop, head });
      labels.push({ x: x - laneW / 2 + 10, y: 204, text: `HEAD ${head} · 独立 K／V`, cls: "head-title", head });
      const miniW = (laneW - 32) / 3;
      ["Q", "K", "V"].forEach((kind, index) => {
        const id = kind.toLowerCase() + "-" + head;
        add(id, x - laneW / 2 + 16 + miniW / 2 + index * miniW, 250, miniW - 5, 72, "W" + kind, `计算 ${kind}_${head}`, `[n,${dh}]`, {
          head, mini: true, formula: `${kind}_${head} = X̂ W${kind},${head}`,
          input: `X̂ [n,${dm}] · W [dmodel,${dh}]`, explain: `第 ${head} 个 Head 的独立 ${kind} 投影。`,
        });
        edge("norm", id, { head });
      });
      add("scores-" + head, x, 390, laneW - 24, 68, "QKᵀ / √d", "QK 点积与缩放", "[n,n]", {
        head, formula: `S${head} = Q${head} K${head}ᵀ / √${dh}`,
        input: `Q${head} [n,${dh}] · K${head} [n,${dh}]`, explain: "每个 Head 独立计算 token 之间的注意力分数。",
      });
      edge("q-" + head, "scores-" + head, { head }); edge("k-" + head, "scores-" + head, { head });
      add("mask-" + head, x, 492, laneW - 24, 68, "Causal mask", "加因果 Mask", "[n,n]", {
        head, formula: "Ŝ[i,j] = j≤i ? S[i,j] : −∞", input: `S${head} [n,n]`, explain: "遮蔽未来 token，保留因果自回归约束。",
      });
      edge("scores-" + head, "mask-" + head, { head });
      add("softmax-" + head, x, 594, laneW - 24, 68, "Softmax", "每行注意力归一化", "[n,n]", {
        head, formula: `A${head} = softmax(Ŝ${head})`, input: `Ŝ${head} [n,n]`, explain: "每个 query 行独立归一化为注意力权重。",
      });
      edge("mask-" + head, "softmax-" + head, { head });
      add("av-" + head, x, avY, laneW - 24, 68, "AV", "注意力权重乘 V", `[n,${dh}]`, {
        head, formula: `C${head} = A${head} V${head}`, input: `A${head} [n,n] · V${head} [n,${dh}]`,
        explain: "V 沿虚线旁路保留，与 Softmax 权重在这里汇合。",
      });
      edge("softmax-" + head, "av-" + head, { head }); edge("v-" + head, "av-" + head, { head, bypass: true }); edge("av-" + head, "concat", { head });
    });
    label(cx, concatY - 45, `C₀ … C${Hq - 1} · 按特征维拼接`, "small-label", "middle");
  } else if (state.arch === "gqa") {
    avY = 920; concatY = 1060;
    const laneTop = 190, laneBottom = 995;
    regions.push({ type: "shared", x: 28, y: 326, w: W - 56, h: 180 });
    label(42, 348, "SHARED KV GROUPS / Hkv = " + state.Hkv, "cache-tag");
    centers.forEach((x, head) => {
      const currentGroup = group(head);
      regions.push({ type: "lane", x: x - laneW / 2, y: laneTop, w: laneW, h: laneBottom - laneTop, head });
      labels.push({ x: x - laneW / 2 + 10, y: 204, text: `HEAD ${head} → GROUP ${currentGroup}`, cls: "head-title", head });
      add("q-" + head, x, 250, laneW - 40, 70, `X̂W_Q,${head}`, "独立 Query 投影", `[n,${dh}]`, {
        head, formula: `Q${head} = X̂ W_Q,${head}`, input: `X̂ [n,${dm}] · W_Q,${head} [dmodel,${dh}]`,
        explain: `GQA 不共享 Query。Q${head} 仍由本 Head 独立计算。`,
      });
      edge("norm", "q-" + head, { head });
      add("scores-" + head, x, 570, laneW - 24, 68, `Q${head}K${currentGroup}ᵀ`, "Q 与共享 K 点积", "[n,n]", {
        head, shared: true, formula: `S${head} = Q${head} K_G${currentGroup}ᵀ / √${dh}`,
        input: `Q${head} [n,${dh}] · K_G${currentGroup} [n,${dh}]`, explain: `Q${head} 读取第 ${currentGroup} 组共享 Key，而不是自己的独立 Key。`,
      });
      edge("q-" + head, "scores-" + head, { head }); edge("k-g" + currentGroup, "scores-" + head, { head, shared: true });
      add("mask-" + head, x, 680, laneW - 24, 68, "Causal mask", "加因果 Mask", "[n,n]", {
        head, formula: "Ŝ = causal_mask(S)", input: `S${head} [n,n]`, explain: "共享 K／V 不改变因果 Mask。",
      });
      edge("scores-" + head, "mask-" + head, { head });
      add("softmax-" + head, x, 790, laneW - 24, 68, "Softmax", "每行注意力归一化", "[n,n]", {
        head, formula: `A${head} = softmax(Ŝ${head})`, input: `Ŝ${head} [n,n]`, explain: "同组 Head 的 Q 不同，因此 Attention 权重仍可不同。",
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
        mini: true, shared: true, cache: true, formula: `K_G${currentGroup} = X̂ W_K,G${currentGroup}`,
        input: `X̂ [n,${dm}]`, explain: `该 Key 由第 ${currentGroup} 组的全部 Query Head 共同读取。`,
      });
      add("v-g" + currentGroup, gx + miniW / 2 + 3, 410, miniW, 66, "V_G" + currentGroup, "共享 Value 投影", `[n,${dh}]`, {
        mini: true, shared: true, cache: true, formula: `V_G${currentGroup} = X̂ W_V,G${currentGroup}`,
        input: `X̂ [n,${dm}]`, explain: `该 Value 由第 ${currentGroup} 组的所有 Query Head 共同读取。`,
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
      custom.push({ kind: "bus", tone: "v", nodes: ["v-g" + currentGroup, ...memberHeads.map((head) => "av-" + head)], d: `M ${v.x + 12} 443 V 478 M ${Math.min(v.x + 12, ...vc)} 478 H ${Math.max(v.x + 12, ...vc)} ` + vc.map((x) => `M ${x} 478 V 920`).join(" ") });
    }
  } else {
    avY = 1390; concatY = 1540;
    const laneTop = 470, laneBottom = 1460;
    regions.push({ type: "shared", x: cx - 302, y: 160, w: 604, h: 170 });
    label(cx - 286, 184, "SHARED COMPRESSED CACHE", "cache-tag");
    add("dkv", cx - 145, 240, 250, 72, "X̂W_DKV", "KV 联合下投影 → cKV", `[n,${state.dc}]`, {
      shared: true, cache: true, formula: "cKV = X̂ W_DKV", input: `X̂ [n,${dm}] · W_DKV [dmodel,${state.dc}]`,
      explain: "K／V 信息被联合压缩到低维 cKV；跨 token 缓存这份 latent。",
    });
    add("kr", cx + 145, 240, 250, 72, "RoPE(X̂W_KR)", "共享解耦 RoPE Key", `[n,${state.dr}]`, {
      shared: true, cache: true, formula: "kR = RoPE(X̂ W_KR)", input: `X̂ [n,${dm}] · W_KR [dmodel,${state.dr}]`,
      explain: "解耦 RoPE Key 单独生成并缓存，不能从 cKV 的维度中漏掉。",
    });
    edge("norm", "dkv", { shared: true }); edge("norm", "kr", { shared: true, tone: "rope" });
    centers.forEach((x, head) => {
      regions.push({ type: "lane", x: x - laneW / 2, y: laneTop, w: laneW, h: laneBottom - laneTop, head });
      labels.push({ x: x - laneW / 2 + 10, y: 492, text: `HEAD ${head} · 共享 cKV`, cls: "head-title", head });
      const miniW = (laneW - 36) / 2;
      add("qc-" + head, x - miniW / 2 - 3, 530, miniW, 68, "Q_C," + head, "内容 Query 投影", `[n,${dh}]`, {
        head, mini: true, formula: `Q_C,${head} = X̂ W_QC,${head}`, input: `X̂ [n,${dm}]`, explain: "当前 Head 的内容 Query 部分。",
      });
      add("qr-" + head, x + miniW / 2 + 3, 530, miniW, 68, "Q_R," + head, "RoPE Query 投影", `[n,${state.dr}]`, {
        head, mini: true, formula: `Q_R,${head} = RoPE(X̂ W_QR,${head})`, input: `X̂ [n,${dm}]`, explain: "当前 Head 的解耦 RoPE Query 部分，不进入 KV Cache。",
      });
      edge("norm", "qc-" + head, { head }); edge("norm", "qr-" + head, { head });
      add("kc-" + head, x - miniW / 2 - 3, 700, miniW, 68, "cW_UK," + head, "从 cKV 还原内容 Key", `[n,${dh}]`, {
        head, mini: true, shared: true, formula: `K_C,${head} = cKV W_UK,${head}`,
        input: `cKV [n,${state.dc}] · W_UK,${head} [dc,${dh}]`, explain: "从同一 cKV 用本 Head 的上游矩阵还原内容 Key。",
      });
      add("v-" + head, x + miniW / 2 + 3, 700, miniW, 68, "cW_UV," + head, "从 cKV 还原 Value", `[n,${dh}]`, {
        head, mini: true, shared: true, formula: `V${head} = cKV W_UV,${head}`,
        input: `cKV [n,${state.dc}] · W_UV,${head} [dc,${dh}]`, explain: "Value 也来自共享 latent，但上游投影按 Head 独立。",
      });
      edge("dkv", "kc-" + head, { head, shared: true }); edge("dkv", "v-" + head, { head, shared: true });
      add("rope-" + head, x, 850, laneW - 24, 68, "Concat + RoPE", "拼成完整 Q／K", `[n,${dh + state.dr}]`, {
        head, formula: `Q${head}=[Q_C;Q_R] · K${head}=[K_C;kR]`,
        input: `Q_C/K_C [n,${dh}] + Q_R/kR [n,${state.dr}]`, explain: "各 Head 的内容部分与共享 kR 拼接，形成完整注意力 Q／K。",
      });
      edge("qc-" + head, "rope-" + head, { head }); edge("qr-" + head, "rope-" + head, { head }); edge("kc-" + head, "rope-" + head, { head }); edge("kr", "rope-" + head, { head, shared: true, tone: "rope" });
      add("scores-" + head, x, 990, laneW - 24, 68, "QKᵀ / √d", "完整 QK 点积", "[n,n]", {
        head, formula: `S${head} = Q${head} K${head}ᵀ / √${dh + state.dr}`,
        input: `Q,K [n,${dh + state.dr}]`, explain: "在内容维与 RoPE 维拼接后的表示上计算分数。",
      });
      edge("rope-" + head, "scores-" + head, { head });
      add("mask-" + head, x, 1110, laneW - 24, 68, "Causal mask", "加因果 Mask", "[n,n]", {
        head, formula: "Ŝ = causal_mask(S)", input: `S${head} [n,n]`, explain: "MLA 仍使用因果 Mask。",
      });
      edge("scores-" + head, "mask-" + head, { head });
      add("softmax-" + head, x, 1230, laneW - 24, 68, "Softmax", "每行注意力归一化", "[n,n]", {
        head, formula: `A${head} = softmax(Ŝ${head})`, input: `Ŝ${head} [n,n]`, explain: "每个 query 行独立归一化为注意力权重。",
      });
      edge("mask-" + head, "softmax-" + head, { head });
      add("av-" + head, x, avY, laneW - 24, 68, "AV", "注意力权重乘 V", `[n,${dh}]`, {
        head, formula: `C${head} = A${head} V${head}`, input: `A${head} [n,n] · V${head} [n,${dh}]`, explain: "Value 来自本 Head 对共享 cKV 的临时还原。",
      });
      edge("softmax-" + head, "av-" + head, { head }); edge("v-" + head, "av-" + head, { head, bypass: true }); edge("av-" + head, "concat", { head });
    });
    label(cx, concatY - 45, `C₀ … C${Hq - 1} · 按特征维拼接`, "small-label", "middle");
    const dkvNode = nodes.find((node) => node.id === "dkv")!, krNode = nodes.find((node) => node.id === "kr")!;
    const qXs = nodes.filter((node) => node.id.startsWith("qc-") || node.id.startsWith("qr-")).map((node) => node.x);
    const kvChannels = centers.map((x) => x + (dkvNode.x < x ? -1 : 1) * (laneW / 2 + 36));
    const krChannels = centers.map((x) => x + (krNode.x < x ? -1 : 1) * (laneW / 2 + 72));
    custom.push({ kind: "bus", tone: "kv", nodes: ["dkv", ...nodes.filter((node) => node.id.startsWith("kc-") || node.id.startsWith("v-")).map((node) => node.id)], d: `M ${dkvNode.x} 276 V 360 ` + kvChannels.map((x) => `M ${dkvNode.x} 360 H ${x} V 640`).join(" ") });
    custom.push({ kind: "bus", tone: "rope", nodes: ["kr", ...centers.map((_, head) => "rope-" + head)], d: `M ${krNode.x} 276 V 395 ` + krChannels.map((x) => `M ${krNode.x} 395 H ${x} V 834`).join(" ") });
    custom.push({ kind: "bus", tone: "q", nodes: ["norm", ...nodes.filter((node) => node.id.startsWith("qc-") || node.id.startsWith("qr-")).map((node) => node.id)], d: `M ${cx} 125 V 435 M ${Math.min(...qXs)} 435 H ${Math.max(...qXs)}` });
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
    explain: "加入的是步骤 18 的完整输出 R = X + P，不是原始 X；再与 MLP 输出 M 相加。",
  });
  edge("linear2", "mlpres");
  custom.push({ kind: "residual", nodes: ["residual", "mlpres"], d: `M ${cx - 36} ${concatY + 240} V ${concatY + 258} H 20 V ${concatY + 780} H ${cx - 125}`, labelX: 28, labelY: concatY + 218, label: "R = X + P · 旁路" });
  return { nodes, edges, regions, labels, custom, W, H: concatY + 870, dm, group, laneW };
}

export function attentionEdgePath(graph: AttentionGraph, state: AttentionState, edge: GraphEdge, a: GraphNode, b: GraphNode): string {
  const y1 = a.y + a.h / 2, y2 = b.y - b.h / 2, mid = Math.round((y1 + y2) / 2);
  const channel = (offset: number, sourceX = a.x) => {
    const center = state.arch === "mla" && (b.id.startsWith("kc-") || b.id.startsWith("qc-")) ? b.x + b.w / 2 + 3
      : state.arch === "mla" && (b.id.startsWith("v-") || b.id.startsWith("qr-")) ? b.x - b.w / 2 - 3 : b.x;
    const side = sourceX < center ? -1 : 1;
    return { side, x: center + side * (graph.laneW / 2 + offset) };
  };
  if (state.arch === "mha" && b.id.startsWith("scores-") && (a.id.startsWith("q-") || a.id.startsWith("k-"))) return `M ${a.x} ${y1} V ${y2}`;
  if (state.arch === "gqa") {
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
  if (state.arch === "mla") {
    if (edge.a === "norm" && (edge.b === "dkv" || edge.b === "kr")) {
      const sourceX = a.x + (edge.b === "dkv" ? -36 : 36);
      return `M ${sourceX} ${y1} V 146 H ${b.x} V ${y2}`;
    }
    if (edge.a === "norm" && (b.id.startsWith("qc-") || b.id.startsWith("qr-"))) return `M ${b.x} 435 V ${y2}`;
    if (edge.a === "dkv" && (b.id.startsWith("kc-") || b.id.startsWith("v-"))) {
      const route = channel(36);
      return `M ${route.x} 640 H ${b.x} V ${y2}`;
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
  }
  if (edge.bypass) {
    const side = state.arch === "mla" ? (b.x < graph.W / 2 ? -1 : 1) : 1;
    const channelX = b.x + side * (graph.laneW / 2 + (state.arch === "mla" ? 36 : 32));
    if (state.arch === "mla") return `M ${a.x} ${y1} V 770 H ${channelX} V ${b.y} H ${b.x + side * b.w / 2}`;
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

export function fullAttentionEdgePath(graph: AttentionGraph, state: AttentionState, edge: GraphEdge, a: GraphNode, b: GraphNode): string {
  const direct = attentionEdgePath(graph, state, edge, a, b), y2 = b.y - b.h / 2;
  if (state.arch === "gqa") {
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
  if (state.arch === "mla") {
    if (edge.a === "norm" && (b.id.startsWith("qc-") || b.id.startsWith("qr-"))) return `M ${graph.W / 2} 125 V 435 H ${b.x} V ${y2}`;
    if (edge.a === "dkv" && (b.id.startsWith("kc-") || b.id.startsWith("v-"))) {
      const center = b.id.startsWith("kc-") ? b.x + b.w / 2 + 3 : b.x - b.w / 2 - 3;
      const side = a.x < center ? -1 : 1, channelX = center + side * (graph.laneW / 2 + 36);
      return `M ${a.x} 276 V 360 H ${channelX} V 640 H ${b.x} V ${y2}`;
    }
    if (edge.a === "kr" && b.id.startsWith("rope-")) {
      const side = a.x < b.x ? -1 : 1, channelX = b.x + side * (graph.laneW / 2 + 72);
      return `M ${a.x} 276 V 395 H ${channelX} V ${b.y - 16} H ${b.x + side * b.w / 2}`;
    }
  }
  return direct;
}
