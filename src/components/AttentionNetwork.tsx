import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import {
  ATTENTION_ARCH_ORDER,
  ATTENTION_ARCHITECTURES,
  INITIAL_ATTENTION_STATE,
  attentionCacheEntries,
  attentionEdgePath,
  buildAttentionGraph,
  fullAttentionEdgePath,
  isLatentAttention,
  isLinearAttention,
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
  const className = `node ${node.kind ? `node-${node.kind}` : ""} ${selected ? "is-selected" : ""} ${related ? "is-related" : ""} ${dim ? "is-dim" : ""}`;
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
          {node.badge && !node.cache && <text className="node-badge" x={node.x} y={y + 65} textAnchor="middle">{node.badge}</text>}
        </>
      ) : (
        <>
          <circle className="index-circle" cx={x + 20} cy={y + 20} r={12} />
          <text className="node-index" x={x + 20} y={y + 23} textAnchor="middle">{String(node.index + 1).padStart(2, "0")}</text>
          <text className="node-short" x={x + 39} y={y + 24}>{node.short}</text>
          <text className="node-title" x={x + 12} y={y + 45}>{node.title}</text>
          <text className="node-shape" x={x + 12} y={y + 61}>{node.shape}</text>
          {node.badge && <text className="node-badge" x={x + node.w - 8} y={y + 15} textAnchor="end">{node.badge}</text>}
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
  const meta = ATTENTION_ARCHITECTURES[state.arch];
  const latent = isLatentAttention(state.arch), linear = isLinearAttention(state.arch);

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
    update({ arch, selected: isLatentAttention(arch) ? "dkv" : isLinearAttention(arch) ? "state0-0" : "norm" });
  }

  function selectNode(id: string) {
    const node = byId[id];
    if (!node) return;
    update({ selected: id, head: node.head ?? state.head });
  }

  function selectHead(head: number) {
    update({ head, selected: latent ? "qc-" + head : linear ? "out-" + head : "q-" + head });
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

  const subtitle = linear
    ? `${state.Hq} 个 Head · 每 Head 状态 ${state.dh}×${state.dh} · ${state.arch === "kda" ? "逐通道" : "标量"}遗忘门`
    : latent
      ? `${state.Hq} 个 Head · 共享 cKV[${state.dc}] ＋ kR[${state.dr}]${state.arch === "dsa" ? ` · Top-${state.topk}` : ""}`
      : state.arch === "mha"
        ? `${state.Hq} 个 Head 并行 · 每 Head 独立 K／V`
        : `${state.Hq} 个 Q Head · ${state.Hkv} 组共享 K／V · ${state.Hq / state.Hkv} Q/组`;
  const currentGroup = graph.group(state.head);
  const cache = attentionCacheEntries(state.arch, state);
  const mapping = state.arch === "mha"
    ? `Q${state.head} 使用独立 K${state.head}、V${state.head}。`
    : state.arch === "mqa"
      ? `Q${state.head} 与全部 ${state.Hq} 个 Query Head 共享同一组 K_G0、V_G0。`
      : state.arch === "gqa"
        ? `Q${state.head} → GROUP ${currentGroup}，与 ${state.Hq / state.Hkv - 1} 个其他 Q 共享 K／V。`
        : latent
          ? `Q${state.head} → 共享 cKV，经 W_UK,${state.head}／W_UV,${state.head} 临时还原；另拼接共享 kR。${state.arch === "dsa" ? ` 主注意力仅读取 Top-${state.topk} 入选位置。` : ""}`
          : `Head ${state.head} 使用独立 S[${state.dh},${state.dh}]；q 负责读出，k 负责寻址，v 提供写入内容。`;
  const groupOptions = Array.from({ length: state.Hq }, (_, index) => index + 1)
    .filter((value) => state.Hq % value === 0 && (state.Hq === 1 || value < state.Hq));
  const cacheRows = ATTENTION_ARCH_ORDER.map((arch) => {
    const value = attentionCacheEntries(arch, state);
    const stateElements = state.Hq * state.dh * state.dh;
    return {
      arch,
      value,
      barValue: value ?? stateElements,
      text: value === null
        ? `无 KV · 全层定长状态 Hq·d² = ${stateElements}`
        : arch === "mha"
          ? `2·${state.Hq}·${state.dh} = ${value}`
          : arch === "mqa"
            ? `2·${state.dh} = ${value}`
            : arch === "gqa"
              ? `2·${state.Hkv}·${state.dh} = ${value}`
              : `${state.dc}+${state.dr} = ${value}${arch === "dsa" ? "（+索引器）" : ""}`,
    };
  });
  const cacheMax = Math.max(...cacheRows.map((row) => row.barValue), 1);

  return (
    <div className="attention-network-page">
      <section className="intro">
        <div>
          <div className="eyebrow">TRANSFORMER BLOCK / DYNAMIC DAG</div>
          <h1>注意力计算网络，随参数重构。</h1>
          <p>沿用实验室的串行主干、并行 Head、分支与残差连接，对照 Softmax 注意力、低秩压缩、稀疏选择和线性注意力。修改结构或超参数，节点、连线、公式与缓存口径会一起变化。</p>
        </div>
        <div className="baseline"><b>初始状态 = 当前实验室</b><br />Hq = 2 · Hkv = 2 · dhead = 4<br />dmodel = 8 · MHA</div>
      </section>

      <section className="controls" aria-label="注意力结构参数">
        <div className="segmented" role="tablist" aria-label="注意力结构">
          {ATTENTION_ARCH_ORDER.map((arch) => (
            <button type="button" role="tab" aria-selected={state.arch === arch} data-arch={arch} key={arch} className={state.arch === arch ? "is-active" : ""} onClick={() => selectArchitecture(arch)}>{ATTENTION_ARCHITECTURES[arch].label}</button>
          ))}
        </div>
        <label className="field"><span>Query Heads · Hq</span><select id="attention-hq" value={state.Hq} onChange={(event) => update({ Hq: Number(event.target.value), head: 0, selected: "norm" })}>{[1, 2, 4, 8].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
        {!latent && !linear && <label className="field" id="hkvField"><span>KV Groups · Hkv</span><select id="attention-hkv" value={state.Hkv} disabled={state.arch !== "gqa"} onChange={(event) => update({ Hkv: Number(event.target.value), selected: "norm" })}>{(state.arch === "gqa" ? groupOptions : [state.Hkv]).map((value) => <option value={value} key={value}>{value} 组 · {state.Hq / value} Q/组</option>)}</select></label>}
        <label className="field"><span>Head 维度 · dhead</span><select id="attention-dh" value={state.dh} onChange={(event) => update({ dh: Number(event.target.value) })}>{[2, 4, 8].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
        {["mha", "mqa", "gqa"].includes(state.arch) && <label className="field" id="sparsityField"><span>稀疏模式</span><select id="attention-sparsity" value={state.sparsity} onChange={(event) => update({ sparsity: event.target.value as AttentionState["sparsity"], selected: "mask-" + state.head })}><option value="dense">Dense · Causal</option><option value="block">Block Sparse · BSA</option><option value="sliding">Sliding Window</option></select></label>}
        {latent && <label className="field" id="dcField"><span>Latent 维度 · dc</span><select id="attention-dc" value={state.dc} onChange={(event) => update({ dc: Number(event.target.value) })}>{[4, 8, 16].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>}
        {latent && <label className="field" id="drField"><span>RoPE 维度 · dR</span><select id="attention-dr" value={state.dr} onChange={(event) => update({ dr: Number(event.target.value) })}>{[2, 4].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>}
        {state.arch === "dsa" && <label className="field" id="topkField"><span>DSA Top-k</span><select id="attention-topk" value={state.topk} onChange={(event) => update({ topk: Number(event.target.value), selected: "topk" })}>{[2, 3].map((value) => <option value={value} key={value}>{value} 个位置 / query</option>)}</select></label>}
        <label className="check"><input id="dimOthers" type="checkbox" checked={state.dim} onChange={(event) => update({ dim: event.target.checked })} />淡化其他 Head</label>
        <div className="zoom"><button type="button" id="zoomOut" aria-label="缩小" onClick={() => setZoom(effectiveZoom - 0.1)}>−</button><output id="zoomOutLabel">{state.zoom === null ? "自动" : Math.round(effectiveZoom * 100) + "%"}</output><button type="button" id="zoomIn" aria-label="放大" onClick={() => setZoom(effectiveZoom + 0.1)}>＋</button><button type="button" id="fit" title="适应当前窗口" onClick={() => update({ zoom: null })}>适应</button></div>
        <button type="button" className="reset" id="reset" onClick={() => setState({ ...INITIAL_ATTENTION_STATE })}>恢复 2 Head MHA</button>
      </section>

      <main className="workspace">
        <section className="network-panel">
          <div className="network-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="3" y="3" width="6" height="5" /><rect x="15" y="3" width="6" height="5" /><rect x="9" y="16" width="6" height="5" /><path d="M6 8v4h12V8M12 12v4" /></svg>
            <div><strong>{meta.title}</strong><span id="networkSubtitle">{subtitle}</span></div>
            <code id="nodeCount">{graph.nodes.length} NODES</code>
          </div>
          <div className="network-scroll" id="networkScroll" ref={scrollRef}>
            <svg id="network" role="img" aria-label={`${meta.label} 注意力计算网络`} viewBox={`0 0 ${graph.W} ${graph.H}`} width={graph.W * effectiveZoom} height={graph.H * effectiveZoom}>
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
                const toneClass = edge.tone === "rope" ? "rope-shared" : edge.tone === "index" ? "index-edge" : edge.tone === "gate" ? "gate-edge" : edge.tone === "loop" ? "loop-edge" : "";
                return <path key={`${edge.a}-${edge.b}`} className={`edge ${edge.shared ? "shared" : ""} ${toneClass} ${edge.bypass ? "v-bypass" : ""} ${dim ? "is-dim" : ""}`} data-edge={`${edge.a}→${edge.b}`} d={attentionEdgePath(graph, state, edge, a, b)} />;
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
                const toneClass = edge.bypass ? "v-bypass v-hot" : edge.tone === "rope" ? "rope-hot" : edge.tone === "index" ? "index-hot" : edge.tone === "gate" ? "gate-hot" : edge.tone === "loop" ? "loop-hot" : edge.shared ? "kv-hot" : "q-hot";
                return <path key={`hot-${edge.a}-${edge.b}`} className={`edge ${toneClass} is-hot`} data-highlight-route={`${edge.a}→${edge.b}`} d={fullAttentionEdgePath(graph, state, edge, a, b)} />;
              })}
              {graph.nodes.map((node) => <NetworkNode key={node.id} node={node} selected={selectedId === node.id} related={isRelated(node)} dim={state.dim && node.head !== undefined && node.head !== state.head} onSelect={selectNode} />)}
            </svg>
          </div>
          <div className="network-legend"><span><i className="current" />选中计算</span><span><i className="parallel" />并行 Head</span>{!linear && <span><i className="shared" />共享 K／V / latent</span>}{linear && <span><i className="state" />定长状态 / 门控</span>}{state.arch === "dsa" && <span><i className="index" />索引与稀疏选择</span>}<span><i className="trunk" />主干旁路</span></div>
        </section>

        <aside className="inspector">
          <section className="inspect-section" id="nodeInspector">
            <div className="kicker">SELECTED NODE / {String(selectedNode.index + 1).padStart(2, "0")}</div>
            <h2>{selectedNode.title}</h2><span className="node-short">{selectedNode.short}</span>
            <div className="formula">{selectedNode.formula}</div>
            <div className="fact"><span>输入</span><code>{selectedNode.input}</code></div>
            <div className="fact"><span>输出</span><code>{selectedNode.output}</code></div>
            <p>{selectedNode.explain}</p>
            {selectedNode.kernel && <div className="note"><b>实现层：</b>{selectedNode.kernel}</div>}
            <div className="head-buttons">{Array.from({ length: state.Hq }, (_, head) => <button type="button" key={head} data-inspect-head={head} className={state.head === head ? "is-active" : ""} onClick={() => selectHead(head)}>Q{head}</button>)}</div>
          </section>
          <section className="inspect-section" id="structureInspector">
            <div className="kicker">CURRENT STRUCTURE</div><h2>{meta.label} · Hq {state.Hq}</h2>
            <p className="architecture-subtitle">{meta.subtitle}</p>
            <div className="stats"><div className="stat"><span>dmodel</span><b>{graph.dm}</b></div><div className="stat"><span>{linear ? "状态 / Head" : "KV groups"}</span><b>{linear ? `${state.dh}²` : latent ? "latent" : state.Hkv}</b></div><div className="stat"><span>{linear ? "KV Cache" : "Cache 元素 / token / 层"}</span><b>{cache ?? "无"}</b></div><div className="stat"><span>网络节点</span><b>{graph.nodes.length}</b></div></div>
            <div className="mapping">{mapping}</div>
            <div className="fact"><span>层级定位</span><code>{meta.position}</code></div>
            <div className="fact"><span>缓存口径</span><code>{meta.cacheFormula}</code></div>
            <div className="fact"><span>代表模型</span><code>{meta.models}</code></div>
            <p>{meta.idea}</p>
            <div className="note"><b>教学要点：</b>{meta.note}</div>
            <div className="note"><b>实现层：</b>{meta.kernel}</div>
          </section>
        </aside>
      </main>

      <div className="reference-sections">
        <section className="inspect-section cache-reference" id="cacheComparison">
          <div className="kicker">PERSISTENT MEMORY / CURRENT PARAMETERS</div>
          <h2>每 token 持久缓存对比</h2>
          <p>单位 = 标量元素数 / token / 层，使用当前 dhead = {state.dh}、dc = {state.dc}、dr = {state.dr}。GDN／KDA 不存 KV；条长度改用全层定长状态 Hq·d² 仅作规模参照，它不是“每 token”缓存。</p>
          <div className="cache-bars">
            {cacheRows.map((row) => <div className={`cache-row ${row.arch === state.arch ? "is-current" : ""}`} key={row.arch} data-cache-arch={row.arch}>
              <span className="cache-name">{ATTENTION_ARCHITECTURES[row.arch].label}</span>
              <span className="cache-track"><i className={row.value === null ? "is-state" : latent && row.arch === state.arch ? "is-latent" : ""} style={{ width: `${Math.max(2, row.barValue / cacheMax * 100)}%` }} /></span>
              <code>{row.text}</code>
            </div>)}
          </div>
          {state.arch === "dsa" && <div className="note">DSA 的 Top-k 只减少主注意力计算，不删除历史 cKV／kR；索引器自身开销未计入上方 dc + dr。</div>}
        </section>

        <section className="inspect-section kernel-reference" id="kernelReference">
          <div className="kicker">ARCHITECTURE ≠ KERNEL</div>
          <h2>实现层对照：kernel 不进图</h2>
          <p>FIA／FAS／SFA／SMLA／BSA／FAG／SFAG／BSAG 等是硬件融合算子或推理后端命名。同一架构的计算图不因 kernel 改名而改变，因此它们不作为并列架构节点。</p>
          <div className="kernel-table-wrap"><table className="kernel-table">
            <thead><tr><th>架构（图层）</th><th>训练 / 计算 kernel</th><th>推理 kernel · 后端</th></tr></thead>
            <tbody>
              <tr><td>MHA / MQA / GQA</td><td>FlashAttention-2/3、cuDNN SDPA</td><td>FA2/FA3、FlashInfer、昇腾 FIA（FusedInferAttentionScore）/ FAS（FlashAttentionScore）；支持 GQA 的变体在部分框架内部记作 FAG</td></tr>
              <tr><td>+ 块稀疏模式</td><td>cuDNN BSA、FA block-sparse varlen</td><td>vLLM / SGLang 块稀疏后端；GQA 组内共享稀疏块（GBSA）对应 BSAG / GBSAG 一类内部命名</td></tr>
              <tr><td>MLA</td><td>FlashMLA（训练侧多用吸收式 MHA 等价实现）</td><td>FlashMLA、vLLM MLA 后端、昇腾 MLA 融合算子</td></tr>
              <tr><td>MLA + DSA</td><td>NSA / DSA 稀疏 kernel + Indexer 前向</td><td>vLLM SFA（Sparse Flash Attention）、昇腾 SMLA / MQSMLA；GQA 支持变体可见 SFAG / SMLAG 一类内部命名</td></tr>
              <tr><td>GDN / KDA</td><td>fla chunkwise_gdn / chunkwise_kda（chunkwise 并行形式）</td><td>fla fused_recurrent 逐步递归 kernel；KDA 为逐通道门控版本</td></tr>
            </tbody>
          </table></div>
        </section>

        <section className="inspect-section reading-reference" id="readingNotes">
          <div className="kicker">READING GUIDE</div><h2>图例与读图须知</h2>
          <div className="reading-grid">
            <ul>
              <li><b>紫色缓存框：</b>跨 token 持久保存的 K／V、cKV 或 kR。</li>
              <li><b>虚线临时还原框：</b>MLA 各 Head 的完整 K／V 只在计算时还原，不占持久缓存。</li>
              <li><b>黄色状态框与递归线：</b>GDN／KDA 的 S 在 token 间滚动更新，尺寸不随 n 增长。</li>
              <li><b>索引与 Top-k：</b>只决定主注意力“算哪些”，不是另一份全量 KV Cache。</li>
            </ul>
            <ul>
              <li>只重构一个 Transformer Block 的注意力部分，并保留 MLP 与两处残差作为图形上下文；n 表示序列长度。</li>
              <li>GDN／KDA 在实际模型中与全注意力层混合堆叠，本页只画单层，并为每个 Head 展开同构状态单元。</li>
              <li>Sparse／Sliding Window 主要改变可见位置与计算量；除非另行驱逐缓存，不能据此认定持久 KV 已减少。</li>
              <li>所有维度均为教学用小数，并非对应真实模型的完整配置。</li>
            </ul>
          </div>
        </section>
      </div>
      <footer className="footer">点击任意节点查看公式、输入输出、解释与实现层说明；点击 Head 标题或右侧 Q 按钮切换检查对象。缩放只改变视图，不改变计算图参数。</footer>
    </div>
  );
}
