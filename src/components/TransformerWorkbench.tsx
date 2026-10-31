import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowDown, ArrowLeft, ArrowRight, Calculator, ChevronDown,
  CirclePause, CirclePlay, MousePointer2, Network, RotateCcw, Shuffle, Sigma,
} from "lucide-react";
import { ProbabilityChart } from "@/components/InferenceVisual";
import {
  OPERATION_COUNT, buildExpansion, buildOperation, fmt,
  type Operation, type OperationTensor,
} from "@/lib/inferenceOperations";
import { chooseCandidate, type Candidate, type ForwardTrace } from "@/lib/tinyTransformer";

const BLUE = [32, 82, 187];
const RED = [229, 67, 46];

function color(value: number, max: number) {
  if (!Number.isFinite(value)) return "transparent";
  const strength = Math.min(Math.abs(value) / Math.max(max, 0.001), 1);
  const rgb = value >= 0 ? BLUE : RED;
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.07 + strength * 0.78})`;
}

function defaultFocus(operation: Operation, selectedToken: number, candidate: Candidate) {
  const output = operation.outputs[0];
  if (["lmhead", "temperature", "topk"].includes(operation.kind) || (operation.kind === "softmax" && operation.group === "输出与自回归")) {
    return { row: 0, col: Math.min(candidate.id, output.data[0].length - 1) };
  }
  if (["sample", "append", "tokenize"].includes(operation.kind)) return { row: 0, col: 0 };
  return { row: Math.min(selectedToken, output.data.length - 1), col: 0 };
}

function TensorView({
  item, selectedToken, focus, onFocus,
}: {
  item: OperationTensor;
  selectedToken: number;
  focus: { row: number; col: number } | null;
  onFocus?: (row: number, col: number) => void;
}) {
  const finite = item.data.flat().filter(Number.isFinite);
  const max = Math.max(...finite.map(Math.abs), 0.001);
  const rows = item.rowLabels ?? item.data.map((_, index) => `row ${index}`);
  const columns = item.colLabels ?? item.data[0].map((_, index) => `d${index}`);
  return (
    <section className={`tensor-block detail-tensor is-${item.role}`}>
      <header><strong>{item.name}</strong><code>{item.shape}</code></header>
      <div className="tensor-scroll" tabIndex={0} aria-label={`${item.name} ${item.shape}，完整数值矩阵，可横向滚动`}>
        <div className="tensor-grid" style={{ gridTemplateColumns: `minmax(88px,1.2fr) repeat(${columns.length},minmax(58px,1fr))` }}>
          <span className="tensor-corner">row \ col</span>
          {columns.map((label, index) => <span className="tensor-column" key={`${label}-${index}`} title={label}>{label}</span>)}
          {item.data.map((row, r) => (
            <div style={{ display: "contents" }} key={`row-${r}`}>
              <span className={`tensor-row-label ${item.role === "output" && selectedToken === r ? "is-selected" : ""}`} title={rows[r]}><span>{rows[r]}</span></span>
              {row.map((value, c) => {
                const masked = item.masked && c > r && !Number.isFinite(value);
                const active = item.role === "output" && focus?.row === r && focus.col === c;
                const content = masked ? "×" : Number.isFinite(value) ? value.toFixed(2) : "−∞";
                const className = `tensor-cell ${active ? "is-active" : ""} ${item.role === "output" && selectedToken === r ? "is-row-active" : ""} ${masked ? "is-masked" : ""}`;
                const style = { backgroundColor: masked ? undefined : color(value, max) };
                const label = `${item.name}[${r},${c}] = ${Number.isFinite(value) ? fmt(value) : "−∞"}`;
                return item.role === "output" && onFocus ? (
                  <button type="button" className={className} style={style} key={`${r}-${c}`} onClick={() => onFocus(r, c)} aria-label={`展开 ${label} 的计算`}>{content}</button>
                ) : <span className={className} style={style} key={`${r}-${c}`} title={label}>{content}</span>;
              })}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TensorFlowZone({ tone, title, items, selectedToken, focus, onFocus, children }: {
  tone: "input" | "parameter" | "output";
  title: string;
  items: OperationTensor[];
  selectedToken: number;
  focus: { row: number; col: number } | null;
  onFocus?: (row: number, col: number) => void;
  children?: ReactNode;
}) {
  return (
    <section className={`flow-tensor-zone is-${tone}`}>
      <header className="flow-zone-heading">
        <span>{tone === "input" ? "INPUT" : tone === "output" ? "OUTPUT" : "PARAMETERS"}</span>
        <strong>{title}</strong>
        <code>{items.map((item) => item.shape).join(" · ")}</code>
      </header>
      {items.length ? <div className="flow-tensor-grid">
        {items.map((item) => <TensorView key={`${item.role}-${item.name}`} item={item} selectedToken={selectedToken} focus={focus} onFocus={onFocus} />)}
      </div> : children}
    </section>
  );
}

function NetworkConnector({ kind = "serial", points = [50], label }: {
  kind?: "serial" | "fork" | "join";
  points?: number[];
  label?: string;
}) {
  const coordinates = points.map((point) => point * 3);
  return (
    <div className={`network-relation is-${kind}`} aria-hidden="true">
      {label && <span>{label}</span>}
      <svg viewBox="0 0 300 42" preserveAspectRatio="none">
        {kind === "serial" && <path d="M150 1 V32" />}
        {kind === "fork" && <>
          <path d="M150 1 V10" />
          {coordinates.map((x) => <path key={x} d={`M150 10 H${x} V32`} />)}
        </>}
        {kind === "join" && <>
          {coordinates.map((x) => <path key={x} d={`M${x} 1 V10 H150`} />)}
          <path d="M150 10 V32" />
        </>}
        {kind === "join" || kind === "serial" ? <path className="relation-arrowhead" d="M145 31 L150 38 L155 31" /> : coordinates.map((x) => <path className="relation-arrowhead" key={`arrow-${x}`} d={`M${x - 5} 31 L${x} 38 L${x + 5} 31`} />)}
      </svg>
    </div>
  );
}

function ResidualConnector({ source, branch }: { source: "X" | "R"; branch: "P" | "M" }) {
  return (
    <div className="network-residual-connector" aria-hidden="true">
      <span className="network-residual-source-label">{source} · 主干旁路</span>
      <span className="network-residual-branch-label">{branch} ↓</span>
      <svg viewBox="0 0 300 54" preserveAspectRatio="none">
        <path className="residual-trunk-line" d="M0 1 V20 H150 V43" />
        <path d="M150 1 V43" />
        <path className="relation-arrowhead" d="M145 36 L150 46 L155 36" />
      </svg>
    </div>
  );
}

export default function TransformerWorkbench({
  trace, candidates, selectedCandidate, onSelectCandidate, onCommitCandidate,
  generatedCount, canGenerate, temperature, topK,
}: {
  trace: ForwardTrace;
  candidates: Candidate[];
  selectedCandidate: Candidate;
  onSelectCandidate: (candidate: Candidate) => void;
  onCommitCandidate: (candidate: Candidate) => void;
  generatedCount: number;
  canGenerate: boolean;
  temperature: number;
  topK: number;
}) {
  const [index, setIndex] = useState(0);
  const [head, setHead] = useState(0);
  const [selectedToken, setSelectedToken] = useState(trace.tokens.length - 1);
  const [focus, setFocus] = useState({ row: trace.tokens.length - 1, col: 0 });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [parametersOpen, setParametersOpen] = useState(false);
  const networkScrollRef = useRef<HTMLDivElement>(null);
  const operations = useMemo(() => Array.from({ length: OPERATION_COUNT }, (_, item) => buildOperation(item, trace, head, temperature, topK, selectedCandidate)), [trace, head, temperature, topK, selectedCandidate]);
  const operation = operations[index];
  const operationGroups = useMemo(() => {
    const groups: { name: string; items: Operation[] }[] = [];
    operations.forEach((item) => {
      const current = groups.at(-1);
      if (current?.name === item.group) current.items.push(item);
      else groups.push({ name: item.group, items: [item] });
    });
    return groups;
  }, [operations]);
  const expansion = useMemo(() => buildExpansion(operation, focus.row, focus.col, trace), [operation, focus, trace]);

  useEffect(() => {
    setFocus(defaultFocus(operation, selectedToken, selectedCandidate));
    setParametersOpen(false);
    // 切换节点时，将数值展开定位到当前 token 的第一个输出元素。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, head]);

  useEffect(() => {
    const scroller = networkScrollRef.current;
    const active = scroller?.querySelector<HTMLElement>(".network-node.is-active");
    if (!scroller || !active) return;
    const scrollerRect = scroller.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    if (activeRect.top < scrollerRect.top + 12 || activeRect.bottom > scrollerRect.bottom - 12) {
      scroller.scrollTo({
        top: scroller.scrollTop + activeRect.top - scrollerRect.top - scroller.clientHeight / 2 + activeRect.height / 2,
        behavior: "smooth",
      });
    }
  }, [index, head]);

  useEffect(() => {
    setSelectedToken((current) => Math.min(current, trace.tokens.length - 1));
  }, [trace.tokens.length]);

  useEffect(() => {
    if (!playing) return;
    if (index >= OPERATION_COUNT - 1) { setPlaying(false); return; }
    const timer = window.setTimeout(() => setIndex((current) => current + 1), 1350 / speed);
    return () => window.clearTimeout(timer);
  }, [playing, index, speed]);

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable) return;
      if (event.code === "Space") { event.preventDefault(); setPlaying((current) => !current); }
      if (event.key === "ArrowRight") { setPlaying(false); setIndex((current) => Math.min(OPERATION_COUNT - 1, current + 1)); }
      if (event.key === "ArrowLeft") { setPlaying(false); setIndex((current) => Math.max(0, current - 1)); }
      if (event.key.toLowerCase() === "r") { setPlaying(false); setIndex(0); }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  function goTo(next: number) {
    setPlaying(false);
    setIndex(Math.max(0, Math.min(OPERATION_COUNT - 1, next)));
  }

  function selectToken(token: number) {
    setSelectedToken(token);
    if (operation.outputs[0].data.length === trace.tokens.length) setFocus((current) => ({ ...current, row: token }));
  }

  function renderNetworkNode(item: Operation, branchHead?: number, compact = false) {
    const isActive = item.index === index && (branchHead === undefined || branchHead === head);
    const headLabel = branchHead === undefined ? "" : `，Head ${branchHead}`;
    const shortLabel = compact ? item.short.replace(/^X̂W_/, "") : item.short;
    return (
      <div className="network-node-wrap" key={`${item.id}${headLabel}`}>
        <button
          type="button"
          className={`network-node ${compact ? "is-compact" : ""} ${isActive ? "is-active" : ""} ${item.index < index ? "is-passed" : ""}`}
          onClick={() => { if (branchHead !== undefined) setHead(branchHead); goTo(item.index); }}
          aria-current={isActive ? "step" : undefined}
          aria-label={`第 ${item.index + 1} 步${headLabel}：${item.title}，输出 ${item.outputs[0]?.shape}`}
        >
          <span className="network-node-index">{String(item.index + 1).padStart(2, "0")}</span>
          <span className="network-node-copy"><strong>{shortLabel}</strong><small>{item.title}</small><code>{item.outputs[0]?.shape}</code></span>
        </button>
      </div>
    );
  }

  function renderNetworkGroup(group: { name: string; items: Operation[] }) {
    const items = group.items;
    if (group.name === "输入与表示") {
      return <div className="network-dag">
        {renderNetworkNode(items[0])}
        <NetworkConnector kind="fork" points={[25, 75]} label="同一序列 · 并行查表" />
        <div className="network-parallel-row is-2">{items.slice(1, 3).map((item) => renderNetworkNode(item))}</div>
        <NetworkConnector kind="join" points={[25, 75]} label="T + P 逐元素相加" />
        {renderNetworkNode(items[3])}
      </div>;
    }
    if (group.name.startsWith("Transformer")) {
      const [norm, q, k, v, scores, mask, softmax, av, concat, projection, residual, mlpNorm, linear1, gelu, linear2, mlpResidual] = items;
      const renderHeadLane = (branchHead: number) => (
        <section className={`network-head-lane ${head === branchHead ? "is-selected" : ""}`} aria-label={`Head ${branchHead} 并行计算分支`} key={branchHead}>
          <button type="button" className="network-head-select" onClick={() => setHead(branchHead)} aria-pressed={head === branchHead}>
            <strong>HEAD {branchHead}</strong><span>{head === branchHead ? "当前检查" : "点击检查"}</span>
          </button>
          <NetworkConnector kind="fork" points={[100 / 6, 50, 500 / 6]} label="Q / K / V 独立投影" />
          <div className="network-parallel-row is-3 network-head-qkv">{[q, k, v].map((item) => renderNetworkNode(item, branchHead, true))}</div>
          <div className="network-attention-branches">
            <div className="network-qk-branch">
              <NetworkConnector kind="join" points={[25, 75]} label="Q, K 点积" />
              {renderNetworkNode(scores, branchHead)}
              <NetworkConnector />
              {renderNetworkNode(mask, branchHead)}
              <NetworkConnector />
              {renderNetworkNode(softmax, branchHead)}
            </div>
            <div className="network-v-bypass"><strong>V</strong><span>保留至 AV</span><i /><ArrowDown /></div>
          </div>
          <NetworkConnector kind="join" points={[100 / 3, 250 / 3]} label="A 与 V 相乘" />
          {renderNetworkNode(av, branchHead)}
        </section>
      );
      return <div className="network-dag">
        <div className="network-phase-label">MULTI-HEAD ATTENTION</div>
        <div className="network-residual-region">
          <div className="network-residual-path">
            <div className="network-trunk-origin"><code>X</code><span>Block 输入 · 保留至残差</span></div>
            {renderNetworkNode(norm)}
            <NetworkConnector kind="fork" points={[25, 75]} label="同一 X̂ · 两个 Head 并行" />
            <div className="network-parallel-row is-2 network-head-parallel">{[0, 1].map(renderHeadLane)}</div>
            <NetworkConnector kind="join" points={[25, 75]} label="C₀ 与 C₁ 按特征维拼接" />
            {renderNetworkNode(concat)}
            <NetworkConnector />
            {renderNetworkNode(projection)}
            <ResidualConnector source="X" branch="P" />
          </div>
          {renderNetworkNode(residual)}
        </div>
        <div className="network-phase-label">MLP + RESIDUAL</div>
        <div className="network-residual-region">
          <div className="network-residual-path">
            <div className="network-trunk-origin"><code>R</code><span>注意力主干 · 保留至残差</span></div>
            {renderNetworkNode(mlpNorm)}
            <NetworkConnector />
            {renderNetworkNode(linear1)}
            <NetworkConnector />
            {renderNetworkNode(gelu)}
            <NetworkConnector />
            {renderNetworkNode(linear2)}
            <ResidualConnector source="R" branch="M" />
          </div>
          {renderNetworkNode(mlpResidual)}
        </div>
      </div>;
    }
    return <div className="network-dag">{items.map((item, itemIndex) => <div className="network-serial-segment" key={item.id}>
      {itemIndex > 0 && <NetworkConnector />}
      {renderNetworkNode(item)}
    </div>)}</div>;
  }

  return (
    <div className="detailed-workbench">
      <header className="detailed-workbench-heading">
        <div><span className="micro-label">FULL NUMERIC FORWARD PASS</span><h2>推理全景 · 43 个实际计算节点</h2><p>左侧滚动查看双 Head 并行、串行与旁路；中间检查输入到输出的数据变化；右侧透视算子内部和逐元素数值展开。</p></div>
        <span className="live-model-badge">2L · 2H · d_model 8</span>
      </header>

      <div className="shared-sample-bar">
        <span>贯穿所有节点的同一个 token 行</span>
        <div className="shared-token-list">{trace.tokens.map((token, tokenIndex) => (
          <button type="button" key={`${token.text}-${tokenIndex}`} className={`${selectedToken === tokenIndex ? "is-selected" : ""} ${tokenIndex >= trace.tokens.length - generatedCount && generatedCount > 0 ? "is-generated" : ""}`} onClick={() => selectToken(tokenIndex)} title={`position ${tokenIndex} · ID ${token.id}`}>{token.text}<small>{tokenIndex}</small></button>
        ))}</div>
        <code>n = {trace.tokens.length}</code>
      </div>

      <div className="inference-panorama-layout">
        <aside className="network-flow-panel" aria-label="完整 Transformer 计算网络，43 个节点定义，两个 Attention Head 并行展开">
          <div className="network-flow-title"><Network /><div><strong>Transformer 计算网络</strong><span>双 Head 并行 · 竖线串行 · 旁路保留</span></div><code>{index + 1}/{OPERATION_COUNT}</code></div>
          <div className="network-flow-groups network-flow-scroll" ref={networkScrollRef} tabIndex={0} aria-label="可滚动的完整计算网络，当前节点会自动定位">
            {operationGroups.map((group, groupIndex) => (
              <section className="network-group" key={group.name}>
                <header className="network-group-label">
                  <div><span>{group.name}</span><small>{group.name.startsWith("Transformer") ? "PRE-NORM DECODER BLOCK" : "DATA FLOW"}</small></div>
                  <code>{group.items.length} 节点</code>
                </header>
                {renderNetworkGroup(group)}
                {groupIndex < operationGroups.length - 1 && <div className="network-group-connector"><ArrowDown /></div>}
              </section>
            ))}
          </div>
          <div className="network-flow-legend"><span><i className="is-current" />当前计算</span><span><i className="is-complete" />已经过</span><span><i className="is-parallel" />并行分支</span><span><i className="is-trunk" />主干旁路</span></div>
        </aside>

        <main className="operation-detail-panel">
          <div className="operation-flow-column">
            <header className="operation-detail-header">
              <div><span>{operation.group} · NODE {String(index + 1).padStart(2, "0")} / {OPERATION_COUNT}</span><h2>{operation.title}</h2><code>{operation.formula}</code></div>
              <div className="detail-step-controls">
                {operation.headRelevant && <div className="head-switch">{[0, 1].map((item) => <button type="button" className={head === item ? "is-active" : ""} key={item} onClick={() => setHead(item)}>Head {item}</button>)}</div>}
                <button type="button" className="icon-button" aria-label="上一个计算" onClick={() => goTo(index - 1)} disabled={index === 0}><ArrowLeft /></button>
                <button type="button" className="icon-button" aria-label="下一个计算" onClick={() => goTo(index + 1)} disabled={index === OPERATION_COUNT - 1}><ArrowRight /></button>
              </div>
            </header>

            <section className="operation-flow-canvas" aria-label={`${operation.title}的输入、参数和输出数据流图`}>
              <TensorFlowZone tone="input" title={index === 0 ? "输入文本 · 按 token 分段" : "输入数据"} items={operation.inputs} selectedToken={selectedToken} focus={null}>
                <div className="flow-text-sequence">{trace.tokens.map((token, tokenIndex) => <span key={`${token.text}-${tokenIndex}`}>{token.raw || token.text}</span>)}</div>
              </TensorFlowZone>

              <div className="flow-transform-arrow">
                <span className="flow-arrow-stem" aria-hidden="true"><i /><ArrowDown /></span>
                {operation.parameters.length ? (
                  <button
                    type="button"
                    className={`transform-arrow-toggle ${parametersOpen ? "is-open" : ""}`}
                    onClick={() => setParametersOpen((current) => !current)}
                    aria-expanded={parametersOpen}
                    aria-controls={`operation-parameters-${index}`}
                  >
                    <span><Calculator />{operation.short}</span>
                    <code>{operation.formula}</code>
                    <strong>{parametersOpen ? "收起参数" : `展开 ${operation.parameters.length} 组参数`}</strong>
                    <ChevronDown />
                  </button>
                ) : (
                  <div className="transform-arrow-toggle is-static">
                    <span><Calculator />{operation.short}</span>
                    <code>{operation.formula}</code>
                    <strong>无参数 · 直接计算</strong>
                  </div>
                )}
                <div className="flow-shape-route">
                  <code>{operation.inputs.map((item) => item.shape).join(" + ")}</code>
                  <ArrowRight />
                  <code>{operation.outputs.map((item) => item.shape).join(" + ")}</code>
                </div>
              </div>

              {parametersOpen && operation.parameters.length > 0 && (
                <div className="flow-parameter-drawer" id={`operation-parameters-${index}`}>
                  <TensorFlowZone tone="parameter" title="箭头中使用的实际参数" items={operation.parameters} selectedToken={selectedToken} focus={null} />
                </div>
              )}

              <div className="flow-output-arrow" aria-hidden="true"><span /><ArrowDown /><code>写回数据流</code></div>
              <TensorFlowZone tone="output" title="输出数据 · 点击元素展开计算" items={operation.outputs} selectedToken={selectedToken} focus={focus} onFocus={(row, col) => setFocus({ row, col })} />
            </section>

            {index === 0 && <div className="token-id-grid detail-token-grid">{trace.tokens.map((token, tokenIndex) => <button type="button" key={`${token.text}-${tokenIndex}`} className={`token-id-cell ${selectedToken === tokenIndex ? "is-selected" : ""} ${token.unknown ? "is-unknown" : ""}`} onClick={() => selectToken(tokenIndex)}><small>position {tokenIndex}</small><strong>{token.text}</strong><span>ID = {token.id}</span>{token.unknown && <i>词表外 → UNK</i>}</button>)}</div>}

            {(index === 40 || index === 41) && <section className="detail-sampling"><div><span>候选分布（仅显示 Top-k）</span><strong>点击候选可改变后续采样节点</strong></div><ProbabilityChart candidates={candidates} selectedId={selectedCandidate.id} onSelect={onSelectCandidate} /></section>}

            {index === 42 && <section className="detail-append-actions"><div><span>本轮选择</span><strong>{selectedCandidate.token}</strong><code>ID {selectedCandidate.id} · p = {(selectedCandidate.probability * 100).toFixed(2)}%</code></div><div className="append-actions"><button type="button" className="button primary" disabled={!canGenerate} onClick={() => onCommitCandidate(selectedCandidate)}>写入并重新计算<ArrowRight /></button><button type="button" className="button secondary" disabled={!canGenerate} onClick={() => onCommitCandidate(chooseCandidate(candidates, true, Math.random()))}><Shuffle />随机采样并写入</button></div>{!canGenerate && <p className="field-error">已达到最大长度或生成 EOS；请运行新的上下文。</p>}</section>}
          </div>

          <aside className="operation-analysis-column" aria-label="当前算子的公式、代码与完整数值展开">
            <header className="analysis-column-heading"><div><span>COMPUTATION INSPECTOR</span><strong>计算透视</strong></div><code>{String(index + 1).padStart(2, "0")} / {OPERATION_COUNT}</code></header>
            <section className="flow-operator-definition">
              <header><Sigma /><span>算子内部</span></header>
              <code className="flow-operator-formula">{operation.formula}</code>
              <p>{operation.explanation}</p>
              <pre><code>{operation.code}</code></pre>
            </section>

            <section className="numeric-expansion-panel" aria-live="polite">
              <header><div><span>选中输出的完整数值展开</span><strong>{expansion.coordinate}</strong></div><output>{expansion.result}</output></header>
              <code className="numeric-expression">{expansion.expression}</code>
              {expansion.terms.length > 0 && <div className="numeric-term-grid">{expansion.terms.map((term, termIndex) => <span key={`${term}-${termIndex}`}>{term}</span>)}</div>}
              <div className="numeric-result-line"><span>按当前算子归约</span><ArrowRight /><strong>{expansion.coordinate} = {expansion.result}</strong></div>
              <p><MousePointer2 />{expansion.note} 点击中间输出的其他单元格可切换。</p>
            </section>
          </aside>

          <div className="operation-controls-column">
            <div className="workbench-controls">
              <button type="button" className="button secondary" onClick={() => goTo(index - 1)} disabled={index === 0}><ArrowLeft />上一个计算</button>
              <button type="button" className="control-play" onClick={() => { if (index >= OPERATION_COUNT - 1) setIndex(0); setPlaying((current) => !current); }}>{playing ? <CirclePause /> : <CirclePlay />}{playing ? "暂停" : "播放计算"}</button>
              <span className="step-counter">{String(index + 1).padStart(2, "0")} / {OPERATION_COUNT}</span>
              <label className="speed-control"><span>速度</span><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={0.7}>0.7×</option><option value={1}>1×</option><option value={1.5}>1.5×</option><option value={2}>2×</option></select></label>
              <button type="button" className="icon-button" aria-label="回到第一个计算" onClick={() => goTo(0)}><RotateCcw /></button>
              <button type="button" className="button primary" onClick={() => goTo(index + 1)} disabled={index === OPERATION_COUNT - 1}>下一个计算<ArrowRight /></button>
            </div>
            <p className="keyboard-hint">← / → 切换计算，空格播放或暂停，R 回到起点。所有数值来自当前上下文的同一次前向计算。</p>
          </div>
        </main>
      </div>
    </div>
  );
}
