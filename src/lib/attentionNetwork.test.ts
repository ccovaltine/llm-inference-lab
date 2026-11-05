import { describe, expect, it } from "vitest";
import {
  ATTENTION_ARCH_ORDER,
  INITIAL_ATTENTION_STATE,
  attentionCacheEntries,
  attentionEdgePath,
  buildAttentionGraph,
  fullAttentionEdgePath,
  normalizeAttentionState,
  type AttentionState,
} from "./attentionNetwork";

function state(patch: Partial<AttentionState>): AttentionState {
  return normalizeAttentionState({ ...INITIAL_ATTENTION_STATE, ...patch });
}

function pathSegments(d: string) {
  const tokens = d.match(/[MHV]|-?\d+(?:\.\d+)?/g) || [];
  const segments: { x1: number; y1: number; x2: number; y2: number }[] = [];
  let x = 0, y = 0, index = 0;
  while (index < tokens.length) {
    const command = tokens[index++];
    if (command === "M") {
      x = Number(tokens[index++]); y = Number(tokens[index++]);
    } else if (command === "H") {
      const nextX = Number(tokens[index++]);
      segments.push({ x1: x, y1: y, x2: nextX, y2: y }); x = nextX;
    } else if (command === "V") {
      const nextY = Number(tokens[index++]);
      segments.push({ x1: x, y1: y, x2: x, y2: nextY }); y = nextY;
    }
  }
  return segments.filter((segment) => segment.x1 !== segment.x2 || segment.y1 !== segment.y2);
}

function crossingNode(d: string, nodes: ReturnType<typeof buildAttentionGraph>["nodes"], allowed: Set<string>) {
  for (const segment of pathSegments(d)) {
    for (const node of nodes) {
      if (allowed.has(node.id)) continue;
      const left = node.x - node.w / 2, right = node.x + node.w / 2;
      const top = node.y - node.h / 2, bottom = node.y + node.h / 2;
      if (segment.x1 === segment.x2) {
        const low = Math.min(segment.y1, segment.y2), high = Math.max(segment.y1, segment.y2);
        if (segment.x1 > left && segment.x1 < right && Math.max(low, top) < Math.min(high, bottom)) return node.id;
      } else {
        const low = Math.min(segment.x1, segment.x2), high = Math.max(segment.x1, segment.x2);
        if (segment.y1 > top && segment.y1 < bottom && Math.max(low, left) < Math.min(high, right)) return node.id;
      }
    }
  }
  return null;
}

describe("dynamic attention architecture graphs", () => {
  it("keeps every architecture graph connected to valid nodes and finite SVG paths", () => {
    for (const arch of ATTENTION_ARCH_ORDER) {
      const current = state({ arch, Hq: 4 });
      const graph = buildAttentionGraph(current);
      const byId = new Map(graph.nodes.map((node) => [node.id, node]));
      expect(byId.size).toBe(graph.nodes.length);
      for (const id of ["norm", "concat", "proj", "residual", "mlpnorm", "linear1", "gelu", "linear2", "mlpres"]) {
        expect(byId.has(id), `${arch} missing ${id}`).toBe(true);
      }
      for (const edge of graph.edges) {
        const a = byId.get(edge.a), b = byId.get(edge.b);
        expect(a, `${arch} missing source ${edge.a}`).toBeDefined();
        expect(b, `${arch} missing target ${edge.b}`).toBeDefined();
        if (!a || !b) continue;
        const direct = attentionEdgePath(graph, current, edge, a, b);
        const full = fullAttentionEdgePath(graph, current, edge, a, b);
        expect(direct).not.toContain("NaN");
        expect(full).not.toContain("NaN");
        expect(direct).not.toContain("undefined");
        expect(full).not.toContain("undefined");
      }
    }
  });

  it("normalizes MHA, MQA and GQA group endpoints without sharing Query", () => {
    expect(state({ arch: "mha", Hq: 8, Hkv: 1 }).Hkv).toBe(8);
    expect(state({ arch: "mqa", Hq: 8, Hkv: 4 }).Hkv).toBe(1);
    expect(state({ arch: "gqa", Hq: 8, Hkv: 8 }).Hkv).toBe(4);
    expect(state({ arch: "gqa", Hq: 4, Hkv: 3 }).Hkv).toBe(2);

    const graph = buildAttentionGraph(state({ arch: "mqa", Hq: 4 }));
    expect(graph.nodes.filter((node) => node.id.startsWith("q-"))).toHaveLength(4);
    expect(graph.nodes.filter((node) => node.id.startsWith("k-g"))).toHaveLength(1);
    expect(graph.nodes.filter((node) => node.id.startsWith("v-g"))).toHaveLength(1);
    expect(graph.group(3)).toBe(0);
  });

  it("changes the mask semantics for dense, block-sparse and sliding-window attention", () => {
    const dense = buildAttentionGraph(state({ arch: "gqa", sparsity: "dense" }));
    const block = buildAttentionGraph(state({ arch: "gqa", sparsity: "block" }));
    const sliding = buildAttentionGraph(state({ arch: "mha", sparsity: "sliding" }));
    expect(dense.nodes.find((node) => node.id === "mask-0")?.short).toBe("Causal Mask");
    expect(block.nodes.find((node) => node.id === "mask-0")?.short).toContain("Block");
    expect(block.nodes.find((node) => node.id === "mask-0")?.explain).toContain("省计算不省缓存");
    expect(sliding.nodes.find((node) => node.id === "mask-0")?.formula).toContain("i−w");
  });

  it("adds causal Top-k selection to every DSA head while retaining the full latent cache", () => {
    const current = state({ arch: "dsa", Hq: 4, topk: 2 });
    const graph = buildAttentionGraph(current);
    expect(graph.nodes.some((node) => node.id === "indexer")).toBe(true);
    expect(graph.nodes.find((node) => node.id === "topk")?.short).toContain("k=2");
    expect(graph.nodes.find((node) => node.id === "dkv")?.cache).toBe(true);
    expect(graph.nodes.find((node) => node.id === "kr")?.cache).toBe(true);
    expect(graph.nodes.find((node) => node.id === "scores-0")?.explain).toContain("不是先计算完整稠密");
    expect(graph.edges.filter((edge) => edge.a === "topk" && edge.b.startsWith("mask-"))).toHaveLength(4);
    expect(attentionCacheEntries("dsa", current)).toBe(current.dc + current.dr);
  });

  it("uses fixed-size Delta Rule states instead of Softmax attention for GDN and KDA", () => {
    for (const arch of ["gdn", "kda"] as const) {
      const graph = buildAttentionGraph(state({ arch, Hq: 4, dh: 4 }));
      expect(graph.nodes.some((node) => node.id.startsWith("scores-"))).toBe(false);
      expect(graph.nodes.some((node) => node.id.startsWith("softmax-"))).toBe(false);
      expect(graph.nodes.filter((node) => node.id.startsWith("state1-"))).toHaveLength(4);
      expect(graph.edges.filter((edge) => edge.a.startsWith("state1-") && edge.b.startsWith("state0-"))).toHaveLength(4);
      expect(attentionCacheEntries(arch, state({ arch }))).toBeNull();
    }
    const gdn = buildAttentionGraph(state({ arch: "gdn" }));
    const kda = buildAttentionGraph(state({ arch: "kda" }));
    expect(gdn.nodes.find((node) => node.id === "write-0")?.formula).toContain("α_t · S");
    expect(kda.nodes.find((node) => node.id === "write-0")?.formula).toContain("diag(α_t)");
    expect(kda.nodes.find((node) => node.id === "a-0")?.shape).toBe("[n,4]");
  });

  it("computes cache sizes from the current toy dimensions", () => {
    const current = state({ Hq: 4, Hkv: 2, dh: 8, dc: 16, dr: 4 });
    expect(attentionCacheEntries("mha", current)).toBe(64);
    expect(attentionCacheEntries("mqa", current)).toBe(16);
    expect(attentionCacheEntries("gqa", current)).toBe(32);
    expect(attentionCacheEntries("mla", current)).toBe(20);
    expect(attentionCacheEntries("dsa", current)).toBe(20);
  });

  it("keeps DSA and linear-attention routes outside unrelated node boxes", () => {
    for (const arch of ["dsa", "gdn", "kda"] as const) {
      for (const Hq of [1, 2, 4, 8]) {
        const current = state({ arch, Hq });
        const graph = buildAttentionGraph(current);
        const byId = new Map(graph.nodes.map((node) => [node.id, node]));
        for (const edge of graph.edges) {
          const a = byId.get(edge.a)!, b = byId.get(edge.b)!;
          const d = attentionEdgePath(graph, current, edge, a, b);
          const full = fullAttentionEdgePath(graph, current, edge, a, b);
          const allowed = new Set([edge.a, edge.b]);
          expect(crossingNode(d, graph.nodes, allowed), `${arch} Hq${Hq}: ${edge.a}→${edge.b}`).toBeNull();
          expect(crossingNode(full, graph.nodes, allowed), `${arch} Hq${Hq} full: ${edge.a}→${edge.b}`).toBeNull();
          if (edge.a === "topk") expect(d).toBe(full);
        }
        for (const path of graph.custom) {
          expect(crossingNode(path.d, graph.nodes, new Set(path.nodes)), `${arch} Hq${Hq} custom: ${path.kind}`).toBeNull();
        }
      }
    }
  });

  it("separates linear-attention input, output and recurrence ports", () => {
    for (const arch of ["gdn", "kda"] as const) {
      const current = state({ arch, Hq: 2 });
      const graph = buildAttentionGraph(current);
      const byId = new Map(graph.nodes.map((node) => [node.id, node]));
      const route = (a: string, b: string) => {
        const edge = graph.edges.find((item) => item.a === a && item.b === b)!;
        return attentionEdgePath(graph, current, edge, byId.get(a)!, byId.get(b)!);
      };
      const alpha = byId.get("a-0")!, write = byId.get("write-0")!;
      expect(route("a-0", "write-0")).toContain(`M ${alpha.x - alpha.w / 2} ${alpha.y}`);
      expect(route("a-0", "write-0")).toContain(`V ${write.y} H ${write.x - write.w / 2}`);
      expect(route("b-0", "write-0")).toContain("V 426");
      expect(route("state0-0", "read-0")).toContain(`${byId.get("state0-0")!.x - 24}`);
      expect(route("state1-0", "out-0")).toContain(`${byId.get("state1-0")!.x - 24}`);
      expect(route("state1-0", "state0-0")).not.toBe(route("state0-0", "write-0"));
      expect(route("k-0", "read-0")).not.toBe(route("k-0", "write-0"));
    }
  });
});
