import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import {
  INITIAL_ATTENTION_STATE,
  attentionEdgePath,
  buildAttentionGraph,
  fullAttentionEdgePath,
  normalizeAttentionState,
  type AttentionArchitecture,
  type AttentionState,
  type GraphNode,
} from "@/lib/attentionNetwork";
import "./AttentionNetwork.css";

function NetworkNode({ node, selected, related, dim, onSelect }: {
  node: GraphNode;
  selected: boolean;
  related: boolean;
  dim: boolean;
  onSelect: (id: string) => void;
}) {
  const x = node.x - node.w / 2, y = node.y - node.h / 2;
  const className = `node ${selected ? "is-selected" : ""} ${related ? "is-related" : ""} ${dim ? "is-dim" : ""}`;
  const choose = () => onSelect(node.id);
  const onKeyDown = (event: KeyboardEvent<SVGGElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  };
  return (
    <g className={className} data-node={node.id} tabIndex={0} role="button" aria-label={node.title} onClick={choose} onKeyDown={onKeyDown}>
      <rect className="node-box" x={x} y={y} width={node.w} height={node.h} rx={3} />
      {node.mini ? (
        <>
          <text className="node-short" x={node.x} y={y + 24} textAnchor="middle">{node.short}</text>
          <text className="node-shape" x={node.x} y={y + 46} textAnchor="middle">{node.shape}</text>
          {node.cache && <text className="cache-tag" x={node.x} y={y + 61} textAnchor="middle">CACHE</text>}
        </>
      ) : (
        <>
          <circle className="index-circle" cx={x + 20} cy={y + 20} r={12} />
          <text className="node-index" x={x + 20} y={y + 23} textAnchor="middle">{String(node.index + 1).padStart(2, "0")}</text>
          <text className="node-short" x={x + 39} y={y + 24}>{node.short}</text>
          <text className="node-title" x={x + 12} y={y + 45}>{node.title}</text>
          <text className="node-shape" x={x + 12} y={y + 61}>{node.shape}</text>
        </>
      )}
    </g>
  );
}

export default function AttentionNetwork() {
  const [state, setState] = useState<AttentionState>({ ...INITIAL_ATTENTION_STATE });
  const [scrollWidth, setScrollWidth] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const graph = useMemo(() => buildAttentionGraph(state), [state]);
  const byId = useMemo(() => Object.fromEntries(graph.nodes.map((node) => [node.id, node])), [graph]);
  const selectedId = byId[state.selected] ? state.selected : "norm";
  const selectedNode = byId[selectedId] || byId.norm;
  const fitZoom = Math.max(0.25, Math.min(1, (scrollWidth - 14) / graph.W));
  const effectiveZoom = state.zoom === null ? fitZoom : state.zoom;

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const updateWidth = () => setScrollWidth(element.clientWidth);
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  function update(patch: Partial<AttentionState>) {
    setState((current) => normalizeAttentionState({ ...current, ...patch }));
  }

  function selectArchitecture(arch: AttentionArchitecture) {
    update({ arch, selected: arch === "mla" ? "dkv" : "norm" });
  }

  function selectNode(id: string) {
    const node = byId[id];
    if (!node) return;
    update({ selected: id, head: node.head ?? state.head });
  }

  function selectHead(head: number) {
    update({ head, selected: state.arch === "mla" ? "qc-" + head : "q-" + head });
  }

  function isRelated(node: GraphNode) {
    return node.id !== selectedId && (
      graph.edges.some((edge) => (edge.a === selectedId && edge.b === node.id) || (edge.b === selectedId && edge.a === node.id)) ||
      graph.custom.some((path) => path.nodes.includes(selectedId) && path.nodes.includes(node.id))
    );
  }

  function setZoom(value: number) {
    update({ zoom: Math.max(0.25, Math.min(1.5, value)) });
  }

  const subtitle = state.arch === "mha"
    ? `${state.Hq} 个 Head 并行 · 每 Head 独立 K／V`
    : state.arch === "gqa"
      ? `${state.Hq} 个 Q Head · ${state.Hkv} 组共享 K／V · ${state.Hq / state.Hkv} Q/组`
      : `${state.Hq} 个 Head · 共享 cKV[${state.dc}] ＋ kR[${state.dr}]`;
  const currentGroup = graph.group(state.head);
  const cache = state.arch === "mla" ? state.dc + state.dr : 2 * state.Hkv * state.dh;
  const mapping = state.arch === "mha"
    ? `Q${state.head} 使用独立 K${state.head}、V${state.head}。`
    : state.arch === "gqa"
      ? `Q${state.head} → GROUP ${currentGroup}，与 ${state.Hq / state.Hkv - 1} 个其他 Q 共享 K／V。`
      : `Q${state.head} → 共享 cKV，经 W_UK,${state.head}／W_UV,${state.head} 还原；另拼接共享 kR。`;
  const groupOptions = Array.from({ length: state.Hq }, (_, index) => index + 1).filter((value) => state.Hq % value === 0);

  return (
    <div className="attention-network-page">
      <section className="intro">
        <div>
          <div className="eyebrow">TRANSFORMER BLOCK / DYNAMIC DAG</div>
          <h1>注意力计算网络，随参数重构。</h1>
          <p>沿用现有实验室的计算网络形状：串行主干、并行 Head、Q／K／V 分支、V 旁路和残差连接。修改结构或超参数，节点、连线与形状会一起变化。</p>
        </div>
        <div className="baseline"><b>初始状态 = 当前实验室</b><br />Hq = 2 · Hkv = 2 · dhead = 4<br />dmodel = 8 · MHA</div>
      </section>

      <section className="controls" aria-label="注意力结构参数">
        <div className="segmented" role="tablist" aria-label="注意力结构">
          {(["mha", "gqa", "mla"] as AttentionArchitecture[]).map((arch) => (
            <button type="button" role="tab" aria-selected={state.arch === arch} data-arch={arch} key={arch} className={state.arch === arch ? "is-active" : ""} onClick={() => selectArchitecture(arch)}>{arch.toUpperCase()}</button>
          ))}
        </div>
        <label className="field"><span>Query Heads · Hq</span><select id="attention-hq" value={state.Hq} onChange={(event) => update({ Hq: Number(event.target.value), head: 0, selected: "norm" })}>{[1, 2, 4, 8].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
        <label className="field" id="hkvField" style={{ visibility: state.arch === "mla" ? "hidden" : "visible" }}><span>KV Groups · Hkv</span><select id="attention-hkv" value={state.Hkv} disabled={state.arch !== "gqa"} onChange={(event) => update({ Hkv: Number(event.target.value), selected: "norm" })}>{groupOptions.map((value) => <option value={value} key={value}>{value} 组 · {state.Hq / value} Q/组</option>)}</select></label>
        <label className="field"><span>Head 维度 · dhead</span><select id="attention-dh" value={state.dh} onChange={(event) => update({ dh: Number(event.target.value) })}>{[2, 4, 8].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
        {state.arch === "mla" && <label className="field" id="dcField"><span>Latent 维度 · dc</span><select id="attention-dc" value={state.dc} onChange={(event) => update({ dc: Number(event.target.value) })}>{[4, 8, 16].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>}
        {state.arch === "mla" && <label className="field" id="drField"><span>RoPE 维度 · dR</span><select id="attention-dr" value={state.dr} onChange={(event) => update({ dr: Number(event.target.value) })}>{[2, 4].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>}
        <label className="check"><input id="dimOthers" type="checkbox" checked={state.dim} onChange={(event) => update({ dim: event.target.checked })} />淡化其他 Head</label>
        <div className="zoom"><button type="button" id="zoomOut" aria-label="缩小" onClick={() => setZoom(effectiveZoom - 0.1)}>−</button><output id="zoomOutLabel">{state.zoom === null ? "自动" : Math.round(effectiveZoom * 100) + "%"}</output><button type="button" id="zoomIn" aria-label="放大" onClick={() => setZoom(effectiveZoom + 0.1)}>＋</button><button type="button" id="fit" title="适应当前窗口" onClick={() => update({ zoom: null })}>适应</button></div>
        <button type="button" className="reset" id="reset" onClick={() => setState({ ...INITIAL_ATTENTION_STATE })}>恢复 2 Head MHA</button>
      </section>

      <main className="workspace">
        <section className="network-panel">
          <div className="network-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="3" y="3" width="6" height="5" /><rect x="15" y="3" width="6" height="5" /><rect x="9" y="16" width="6" height="5" /><path d="M6 8v4h12V8M12 12v4" /></svg>
            <div><strong>Transformer Block · 注意力计算网络</strong><span id="networkSubtitle">{subtitle}</span></div>
            <code id="nodeCount">{graph.nodes.length} NODES</code>
          </div>
          <div className="network-scroll" id="networkScroll" ref={scrollRef}>
            <svg id="network" role="img" aria-label="随参数变化的注意力计算网络" viewBox={`0 0 ${graph.W} ${graph.H}`} width={graph.W * effectiveZoom} height={graph.H * effectiveZoom}>
              <defs>
                <marker id="attention-arrow" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(43% .19 267)" /></marker>
                <marker id="attention-arrowHot" markerWidth="8" markerHeight="8" refX="5.5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(51% .18 261)" strokeWidth="1.4" /></marker>
                <marker id="attention-arrowPurple" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(52% .16 305)" /></marker>
                <marker id="attention-arrowYellow" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(79% .14 82)" /></marker>
                <marker id="attention-arrowTeal" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(48% .09 195)" /></marker>
                <marker id="attention-arrowOrange" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(58% .14 55)" /></marker>
              </defs>
              {graph.regions.slice().sort((a, b) => Number(a.type === "shared") - Number(b.type === "shared")).map((region, index) => <rect key={`${region.type}-${index}`} className={`${region.type === "shared" ? "shared-region" : "lane"} ${region.head === state.head ? "is-selected" : ""}`} x={region.x} y={region.y} width={region.w} height={region.h} rx={2} />)}
              {graph.labels.map((label, index) => (
                <g key={`${label.text}-${index}`}>
                  <text className={label.cls} x={label.x} y={label.y} textAnchor={label.anchor as "start" | "middle" | "end" | undefined}>{label.text}</text>
                  {label.head !== undefined && <g className="head-hit" data-head={label.head} tabIndex={0} role="button" aria-label={`检查 Head ${label.head}`} onClick={() => selectHead(label.head!)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectHead(label.head!); } }}><rect x={label.x - 5} y={label.y - 17} width={graph.W / state.Hq > 200 ? 185 : 170} height={25} /><title>检查 Head {label.head}</title></g>}
                </g>
              ))}
              {graph.edges.map((edge) => {
                const a = byId[edge.a], b = byId[edge.b];
                if (!a || !b) return null;
                const dim = state.dim && edge.head !== undefined && edge.head !== state.head;
                return <path key={`${edge.a}-${edge.b}`} className={`edge ${edge.shared ? "shared" : ""} ${edge.tone === "rope" ? "rope-shared" : ""} ${edge.bypass ? "v-bypass" : ""} ${dim ? "is-dim" : ""}`} data-edge={`${edge.a}→${edge.b}`} d={attentionEdgePath(graph, state, edge, a, b)} />;
              })}
              {graph.custom.map((path, index) => {
                const hot = path.nodes.includes(selectedId);
                if (path.kind === "bus") return <path key={`bus-${index}`} className={`edge bus ${path.tone}-bus`} d={path.d} />;
                if (path.kind === "input") return <path key={`input-${index}`} className={`edge residual ${hot ? "is-hot residual-hot" : ""}`} data-input-route="X→norm" d={path.d} />;
                return <g key={`residual-${index}`}><path className={`edge residual ${hot ? "is-hot residual-hot" : ""}`} data-residual-route={path.label} d={path.d} /><text className="residual-label" x={path.labelX} y={path.labelY}>{path.label}</text></g>;
              })}
              {graph.edges.filter((edge) => selectedId === edge.a || selectedId === edge.b).map((edge) => {
                const a = byId[edge.a], b = byId[edge.b];
                if (!a || !b) return null;
                const toneClass = edge.bypass ? "v-bypass v-hot" : edge.tone === "rope" ? "rope-hot" : edge.shared ? "kv-hot" : "q-hot";
                return <path key={`hot-${edge.a}-${edge.b}`} className={`edge ${toneClass} is-hot`} data-highlight-route={`${edge.a}→${edge.b}`} d={fullAttentionEdgePath(graph, state, edge, a, b)} />;
              })}
              {graph.nodes.map((node) => <NetworkNode key={node.id} node={node} selected={selectedId === node.id} related={isRelated(node)} dim={state.dim && node.head !== undefined && node.head !== state.head} onSelect={selectNode} />)}
            </svg>
          </div>
          <div className="network-legend"><span><i className="current" />选中计算</span><span><i className="parallel" />并行 Head</span><span><i className="shared" />共享 K／V / latent</span><span><i className="trunk" />主干旁路</span></div>
        </section>

        <aside className="inspector">
          <section className="inspect-section" id="nodeInspector">
            <div className="kicker">SELECTED NODE / {String(selectedNode.index + 1).padStart(2, "0")}</div>
            <h2>{selectedNode.title}</h2><span className="node-short">{selectedNode.short}</span>
            <div className="formula">{selectedNode.formula}</div>
            <div className="fact"><span>输入</span><code>{selectedNode.input}</code></div>
            <div className="fact"><span>输出</span><code>{selectedNode.output}</code></div>
            <p>{selectedNode.explain}</p>
            <div className="head-buttons">{Array.from({ length: state.Hq }, (_, head) => <button type="button" key={head} data-inspect-head={head} className={state.head === head ? "is-active" : ""} onClick={() => selectHead(head)}>Q{head}</button>)}</div>
          </section>
          <section className="inspect-section" id="structureInspector">
            <div className="kicker">CURRENT STRUCTURE</div><h2>{state.arch.toUpperCase()} · Hq {state.Hq}</h2>
            <div className="stats"><div className="stat"><span>dmodel</span><b>{graph.dm}</b></div><div className="stat"><span>KV groups</span><b>{state.arch === "mla" ? "latent" : state.Hkv}</b></div><div className="stat"><span>Cache 元素 / token / 层</span><b>{cache}</b></div><div className="stat"><span>网络节点</span><b>{graph.nodes.length}</b></div></div>
            <div className="mapping">{mapping}</div>
            <div className="note">{state.arch === "mha" ? "MHA：Q、K、V 都按 Head 独立；Concat 的宽度为 Hq × dhead。" : state.arch === "gqa" ? "GQA：Q 节点数量保持 Hq；共享 K／V 节点数量变为 Hkv。连线直接显示组与 Head 的映射。" : "MLA：cKV 与 kR 是共享缓存节点；各 Head 的完整 K／V 在计算时临时还原。"}</div>
          </section>
        </aside>
      </main>
      <footer className="footer">原型只重构一个 Transformer Block 的注意力部分，并保留 MLP 与两处残差作为图形上下文。n 表示序列长度；MLA 采用 cKV ＋ 解耦 RoPE Key 的教学形式，临时还原的完整 K／V 不等于持久缓存。</footer>
    </div>
  );
}
