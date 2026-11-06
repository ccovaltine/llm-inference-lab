import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import "./KVCacheReplay.css";

const PROMPT = ["小猫", "坐", "在", "垫子", "上"];
const GENERATED = ["然后", "睡着", "了", "。"];
const ALL = [...PROMPT, ...GENERATED];
const LAST_STEP = GENERATED.length;
const DH = 4;

type Focus = "scores" | "mask" | "softmax" | "av";
type Mode = "uncached" | "cached";
type Selection = { q: number; k: number; d: number };

const FOCUS_INFO: Record<Focus, { title: string; short: string; formula: string; desc: string }> = {
  scores: { title: "QK 点积与缩放", short: "QKᵀ / √d", formula: "S₀ = Q₀K₀ᵀ / √4", desc: "矩阵是注意力分数 S：第 i 行是第 i 个 Query，第 j 列是它读取第 j 个 Key 的匹配分数。" },
  mask: { title: "加因果 Mask", short: "Causal mask", formula: "Ŝ[i,j] = j≤i ? S[i,j] : −∞", desc: "在 S 上遮蔽未来位置。斜纹格不能参与后续 Softmax，防止 token 看见未来。" },
  softmax: { title: "每行注意力归一化", short: "Softmax", formula: "A₀ = softmax(Ŝ₀)", desc: "矩阵是注意力权重 A：每个有效 Query 行的权重之和为 1。" },
  av: { title: "注意力权重乘 V", short: "AV", formula: "C₀ = A₀V₀", desc: "这里是 Head 输出 C：每行 4 个数，是该 Query 对所有历史 V 的加权和。" },
};

const fmt = (value: number) => Number.isFinite(value) ? Math.abs(value) < 0.0005 ? "0.000" : value.toFixed(3) : "−∞";

function vector(kind: "q" | "k" | "v", i: number) {
  return Array.from({ length: DH }, (_, d) => {
    if (kind === "q") return Math.sin((i + 1) * 0.73 + (d + 1) * 0.61) * 1.1 + Math.cos((i + 2) * (d + 1) * 0.17) * 0.25;
    if (kind === "k") return Math.cos((i + 1) * 0.57 + (d + 1) * 0.49) + Math.sin((i + 3) * (d + 1) * 0.13) * 0.22;
    return Math.sin((i + 1) * 0.41 + (d + 1) * 0.83) * 0.85 + (d - 1.5) * 0.08;
  });
}

const Q = ALL.map((_, i) => vector("q", i));
const K = ALL.map((_, i) => vector("k", i));
const V = ALL.map((_, i) => vector("v", i));

function score(i: number, j: number) {
  return Q[i].reduce((sum, value, d) => sum + value * K[j][d], 0) / Math.sqrt(DH);
}

function weights(i: number) {
  const raw = Array.from({ length: i + 1 }, (_, j) => score(i, j));
  const max = Math.max(...raw);
  const exponents = raw.map((value) => Math.exp(value - max));
  const sum = exponents.reduce((a, b) => a + b, 0);
  return ALL.map((_, j) => j <= i ? exponents[j] / sum : 0);
}

const A = ALL.map((_, i) => weights(i));
const context = (i: number, d: number) => A[i].reduce((sum, weight, j) => sum + weight * V[j][d], 0);

function valueAt(focus: Focus, i: number, j: number) {
  if (focus === "scores") return score(i, j);
  if (focus === "mask") return j <= i ? score(i, j) : -Infinity;
  if (focus === "softmax") return A[i][j];
  return context(i, j);
}

function cellStyle(focus: Focus, value: number): CSSProperties {
  if (!Number.isFinite(value)) return {};
  if (focus === "softmax") return { background: `rgba(32,82,187,${(0.06 + Math.max(0, value) * 0.72).toFixed(3)})` };
  const alpha = Math.min(0.68, Math.abs(value) * 0.27 + 0.04);
  return { background: value >= 0 ? `rgba(32,82,187,${alpha.toFixed(3)})` : `rgba(207,104,55,${alpha.toFixed(3)})` };
}

function OperandRows({ kind, data, t, prefill, current }: { kind: "q" | "k" | "v"; data: number[][]; t: number; prefill: boolean; current: number }) {
  return (
    <div className="operand-list">
      {data.slice(0, t).map((values, position) => (
        <div className={`operand-row ${!prefill && position === current ? "current" : ""}`} data-operand-row={kind} data-position={position} key={position}>
          <span className="operand-token">{kind.toUpperCase()}{position} · {ALL[position]}<small>position {position}</small></span>
          <div className="vector">{values.map((value, d) => <i key={d}>{fmt(value)}</i>)}</div>
        </div>
      ))}
    </div>
  );
}

function AttentionMatrix({ mode, focus, rows, t, selected, onSelect }: { mode: Mode; focus: Focus; rows: number[]; t: number; selected: Selection; onSelect: (selection: Selection) => void }) {
  const av = focus === "av";
  const columns = av ? DH : t;
  return (
    <div className="attention-matrix" data-matrix={mode} style={{ gridTemplateColumns: `82px repeat(${columns},minmax(42px,1fr))` }}>
      <span className="m-corner">Q \ {av ? "dim" : "K"}</span>
      {Array.from({ length: columns }, (_, j) => <span className="m-head" key={j}>{av ? `c${j}` : ALL[j]}<small>{av ? "V 维度" : `K${j}`}</small></span>)}
      {rows.map((i) => (
        <div className="matrix-row-contents" style={{ display: "contents" }} key={i}>
          <span className="m-head row">Q{i} · {ALL[i]}</span>
          {Array.from({ length: columns }, (_, j) => {
            const value = valueAt(focus, i, j);
            const masked = !av && j > i && focus !== "scores";
            const isSelected = selected.q === i && (av ? selected.d === j : selected.k === j);
            return (
              <button type="button" className={`m-cell ${masked ? "is-masked" : ""} ${isSelected ? "is-selected" : ""}`} style={masked ? {} : cellStyle(focus, value)} data-q={i} data-k={j} aria-label={`Q${i} ${ALL[i]}，${av ? `输出维度 ${j}` : `K${j} ${ALL[j]}`}，${fmt(value)}`} onClick={() => onSelect({ q: i, k: j, d: j })} key={j}>
                {masked ? "−∞" : fmt(value)}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function NetworkOperator({ focus, currentFocus, label, sub, x, y, onSelect }: { focus: Focus; currentFocus: Focus; label: string; sub: string; x: number; y: number; onSelect: (focus: Focus) => void }) {
  return (
    <g className={`op is-active ${currentFocus === focus ? "is-focused" : ""}`} data-focus={focus} tabIndex={0} role="button" aria-label={`查看${FOCUS_INFO[focus].title}`} aria-pressed={currentFocus === focus} onClick={() => onSelect(focus)} onKeyDown={(event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(focus); }
    }}>
      <rect x={x} y={y} width="220" height="62" rx="3" />
      <text x={x + 110} y={y + 28}>{label}</text>
      <text className="sub" x={x + 110} y={y + 47}>{sub}</text>
    </g>
  );
}

export default function KVCacheReplay() {
  const [step, setStep] = useState(0);
  const [focus, setFocus] = useState<Focus>("scores");
  const [selected, setSelected] = useState<Selection>({ q: 4, k: 4, d: 0 });
  const [playing, setPlaying] = useState(false);
  const [dockOpen, setDockOpen] = useState(() => window.matchMedia("(min-width:1200px)").matches);
  const [viewportAnchor, setViewportAnchor] = useState("shared-computation");
  const [activeSection, setActiveSection] = useState("shared-computation");
  const [contextLength, setContextLength] = useState(4096);
  const stepAnchorRef = useRef<{ id: string; top: number } | null>(null);

  const prefill = step === 0;
  const t = PROMPT.length + step;
  const current = t - 1;
  const rows = prefill ? Array.from({ length: t }, (_, i) => i) : [current];
  const info = FOCUS_INFO[focus];
  const activeSelected = { q: Math.min(selected.q, current), k: Math.min(selected.k, current), d: selected.d };

  useLayoutEffect(() => {
    const anchor = stepAnchorRef.current;
    if (!anchor) return;
    stepAnchorRef.current = null;
    const target = document.getElementById(anchor.id);
    if (!target) return;
    const delta = target.getBoundingClientRect().top - anchor.top;
    if (Math.abs(delta) > 0.5) window.scrollBy({ top: delta, behavior: "instant" });
  }, [step]);

  const goToStep = useCallback((nextStep: number) => {
    const next = Math.max(0, Math.min(LAST_STEP, nextStep));
    if (viewportAnchor !== "shared-computation") {
      const target = document.getElementById(viewportAnchor);
      if (target) stepAnchorRef.current = { id: viewportAnchor, top: target.getBoundingClientRect().top };
    }
    setStep(next);
    const nextCurrent = PROMPT.length + next - 1;
    setSelected({ q: nextCurrent, k: nextCurrent, d: 0 });
  }, [viewportAnchor]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => {
      if (step >= LAST_STEP) setPlaying(false);
      else goToStep(step + 1);
    }, step >= LAST_STEP ? 0 : 1500);
    return () => window.clearTimeout(timer);
  }, [playing, step, goToStep]);

  useEffect(() => {
    const media = window.matchMedia("(min-width:1200px)");
    const update = (event: MediaQueryListEvent) => setDockOpen(event.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => { if (entry.isIntersecting) setActiveSection(entry.target.id); });
    }, { rootMargin: "-12% 0px -60% 0px", threshold: 0 });
    ["shared-computation", "execution-comparison", "numeric-expansion"].forEach((id) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });
    return () => observer.disconnect();
  }, []);

  function selectOperator(next: Focus) {
    setFocus(next);
    setSelected({ q: current, k: current, d: 0 });
  }

  function scrollToSection(id: string) {
    setViewportAnchor(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function stopAndGo(next: number) {
    setPlaying(false);
    goToStep(next);
  }

  const inspectorRow = activeSelected.q;
  const inspectorColumn = focus === "av" ? activeSelected.d : activeSelected.k;
  const inspectorValue = focus === "av" ? context(inspectorRow, inspectorColumn) : valueAt(focus, inspectorRow, inspectorColumn);
  const cacheBytes = contextLength * 32 * 2 * 8 * 128 * 2;

  function renderNeeded(mode: Mode) {
    const cached = mode === "cached";
    const qRows = prefill ? t : 1;
    const kvNewRows = prefill || !cached ? t : 1;
    return (
      <>
        <div className="needed-token"><strong>{prefill ? `整段前缀 ${t} tokens` : cached ? `当前 token“${ALL[current]}”` : "当前 Q + 整段 K／V"}</strong><code>{prefill ? `positions 0…${current}` : `Q position ${current} · K/V ${cached ? `仅 position ${current}` : `positions 0…${current}`}`}</code></div>
        <div className="compute-row q"><strong>Q₀</strong><span>Query 投影{!prefill ? " · 只算当前行" : ""}</span><code>{qRows} 行 × 4 维</code></div>
        <div className="compute-row k"><strong>K₀</strong><span>Key 投影{cached && !prefill ? "；随后追加缓存" : !prefill ? " · 历史也重算" : ""}</span><code>{kvNewRows} 行 × 4 维</code></div>
        <div className="compute-row v"><strong>V₀</strong><span>Value 投影{cached && !prefill ? "；随后追加缓存" : !prefill ? " · 历史也重算" : ""}</span><code>{kvNewRows} 行 × 4 维</code></div>
        <div className="compute-flow">
          {prefill ? <>计算完整因果注意力：<b>{t} 个 Q 行</b></> : <>计算当前一行注意力：<b>1 个 Q 行 × {t} 份 K／V</b></>}
          <br />QKᵀ → Mask → Softmax → AV → Concat → 输出投影
          {!prefill && !cached && <><br /><b>无需重算历史 Q 或历史 QK；它们对当前输出不是必需项。</b><br />朴素整段前向会顺带重算历史行；这里展示生成当前 token 的最小必要计算。</>}
        </div>
      </>
    );
  }

  function renderCache(mode: Mode) {
    const cached = mode === "cached";
    return (
      <>
        <div className="cache-title"><strong>{cached ? "持久缓存内容" : "临时产生，不作为缓存保存"}</strong><code>K[4] + V[4] / token</code></div>
        <div className="cache-columns"><span>token / 状态</span><span>K 向量</span><span>V 向量</span></div>
        {ALL.slice(0, t).map((token, index) => {
          const state = cached ? prefill || index === current ? "write" : "reuse" : "recompute";
          const label = state === "write" ? prefill ? "↓ 写入缓存" : "↓ 本轮追加" : state === "reuse" ? "↗ 已缓存 · 读取" : "↻ 本轮重算";
          return (
            <div className={`cache-row ${state}`} key={index}>
              <span className="cache-state">{label}</span><span className="cache-token">{token}<small>position {index}</small></span>
              <span className="kv-vector k">{K[index].map((value, d) => <i key={d}>{fmt(value)}</i>)}</span>
              <span className="kv-vector v">{V[index].map((value, d) => <i key={d}>{fmt(value)}</i>)}</span>
            </div>
          );
        })}
        <p className="cache-note">{cached ? prefill ? `Prefill 注意力完成，${t} 份 K／V 已写入缓存，供后续 Decode 使用。` : `青色 ${t - 1} 行是此前已经缓存的历史；紫色 1 行是本轮新算并追加的当前 K／V。` : "这些不是“已经缓存的内容”，而是本轮不得不重新计算的 K／V；进入下一轮后仍需全部重算。"}</p>
      </>
    );
  }

  function renderInspector() {
    const i = inspectorRow;
    const j = inspectorColumn;
    let title = "";
    let expression = "";
    let terms: string[] = [];
    let note = "";
    if (focus === "scores") {
      title = `S[${i},${j}] · “${ALL[i]}”关注“${ALL[j]}”`;
      expression = `S[${i},${j}] = (Q${i} · K${j}) / √4 = ${fmt(inspectorValue)}`;
      terms = Q[i].map((value, d) => `q${d}·k${d} = ${fmt(value)}×${fmt(K[j][d])} = ${fmt(value * K[j][d])}`);
      note = "四项乘积相加后再除以 √4 = 2。这里尚未做因果 Mask 与 Softmax。";
    } else if (focus === "mask") {
      title = `Ŝ[${i},${j}] · 因果遮蔽`;
      expression = j <= i ? `j=${j} ≤ i=${i}，保留 S[${i},${j}] = ${fmt(inspectorValue)}` : `j=${j} > i=${i}，未来位置设为 −∞`;
      terms = [`原始分数 S = ${fmt(score(i, j))}`, `Mask 后 Ŝ = ${fmt(inspectorValue)}`];
      note = j <= i ? "该位置在过去或当前，可以进入 Softmax。" : "该位置属于未来，Softmax 后权重为 0。";
    } else if (focus === "softmax") {
      title = `A[${i},${j}] · 注意力权重`;
      expression = `A[${i},${j}] = exp(Ŝ[${i},${j}] − max) / Σ exp(Ŝ[${i},:] − max) = ${fmt(inspectorValue)}`;
      const max = Math.max(...Array.from({ length: i + 1 }, (_, k) => score(i, k)));
      terms = Array.from({ length: i + 1 }, (_, k) => `exp 项 ${k}: ${fmt(Math.exp(score(i, k) - max))}`);
      terms.push(`行权重总和 = ${A[i].reduce((a, b) => a + b, 0).toFixed(3)}`);
      note = "缓存 K 让当前行能重新得到这些权重；缓存中并没有保存这行 A。";
    } else {
      title = `C[${i},${j}] · Head 输出维度 ${j}`;
      expression = `C[${i},${j}] = Σₖ A[${i},k] × V[k,${j}] = ${fmt(inspectorValue)}`;
      terms = Array.from({ length: i + 1 }, (_, k) => `A[${i},${k}]×V[${k},${j}] = ${fmt(A[i][k])}×${fmt(V[k][j])} = ${fmt(A[i][k] * V[k][j])}`);
      note = "这里复用的是全部历史 V；加权结果进入 Concat，再经过输出投影与残差。";
    }
    return { title, expression, terms, note };
  }

  const inspector = renderInspector();

  function renderLane(mode: Mode) {
    const cached = mode === "cached";
    const stats = { newKV: cached ? prefill ? t : 1 : t, reuse: cached && !prefill ? t - 1 : 0, queries: rows.length, pairs: rows.reduce((sum, i) => sum + i + 1, 0) };
    return (
      <article className={`lane ${mode}`} data-mode={mode}>
        <header className="lane-header">
          <span className="eyebrow">{cached ? "② PATH B / PERSISTENT KV CACHE" : "② PATH A / NO PERSISTENT CACHE"}</span>
          <h2>{cached ? "有缓存：新算什么，已有什么？" : "无缓存：本轮需要计算哪些？"}</h2>
          <p data-lane-description={mode}>{prefill ? cached ? "Prefill 同样计算完整提示词，但会把全部 K／V 写入缓存，供后续 Decode 直接读取。" : "Prefill 并行计算全部提示 Q／K／V 和各行因果注意力；K／V 只用本轮，不跨轮保留。" : cached ? "只处理当前 token 的新表示；历史 K／V 从缓存读取，不再重算。" : "当前 Q 只算一行；没有旧 K／V 可读取，所以历史与当前 K／V 都要重算，历史 Q／QK 不必重算。"}</p>
        </header>
        <div className="lane-stats">
          <div className="lane-stat"><b data-stat="newKV">{stats.newKV}</b><span>需新算 K／V 的 token 数</span></div>
          <div className="lane-stat"><b data-stat="reuse">{stats.reuse}</b><span>已缓存可复用</span></div>
          <div className="lane-stat"><b data-stat="queries">{stats.queries}</b><span>需计算 Q 行数</span></div>
          <div className="lane-stat"><b data-stat="pairs">{stats.pairs}</b><span>有效 Q→K 连接</span></div>
        </div>
        <div className="compute-section">
          <div className="compute-title"><h3>{cached ? "本轮必须新算" : "本轮必须产生"}</h3><code data-needed-label>{prefill ? `ALL ${t} TOKENS` : `Q 1 · K/V ${cached ? "NEW 1" : `ALL ${t}`}`}</code></div>
          <div className="needed-panel" data-needed={mode}>{renderNeeded(mode)}</div>
        </div>
        <div className="history-block">
          <div className="compute-title"><h3>{cached ? "缓存中已经存在 + 本轮追加" : "本轮临时 K／V：用完即弃"}</h3><code>{cached ? "LAYER 1 / HEAD 0" : "NO SAVED HISTORY"}</code></div>
          <div className="cache-legend">{cached ? <><span><i className="reuse">↗</i>此前已缓存，本轮读取</span><span><i className="write">↓</i>本轮新算并追加</span></> : <span><i className="recompute">↻</i>本轮重算，不跨轮保留</span>}</div>
          <div className="cache-panel" data-cache={mode}>{renderCache(mode)}</div>
        </div>
        <div className="operator-inspector">
          <div className="compute-title"><h3 data-operator-title>{info.title}</h3><code data-matrix-shape>{focus === "av" ? `[${rows.length},${DH}]` : `[${rows.length},${t}]`}</code></div>
          <div className="operator-note" data-operator-note={mode}><strong>{info.short} · {prefill ? "Prefill" : "Decode"}</strong><code>{info.formula} · Q 行数 {rows.length} · K/V 长度 {t}</code><p>{info.desc}</p></div>
          <div className="matrix-help"><span>行 = Query · 列 = <b data-column-label>{focus === "av" ? "Head 输出维度" : "Key token"}</b></span><span>点击数值展开</span></div>
          <div className="matrix-scroll"><AttentionMatrix mode={mode} focus={focus} rows={rows} t={t} selected={activeSelected} onSelect={setSelected} /></div>
          <p className="matrix-caption" data-matrix-caption={mode}>{focus === "av" ? "每行是 4 维 Head 输出，后续进入 Concat。" : `本轮实际计算 ${stats.pairs} 个有效 Q→K 连接。`}</p>
        </div>
      </article>
    );
  }

  return (
    <div className="kv-replay-page">
      <main className="lab-section">
        <section className="page-intro">
          <div>
            <span className="eyebrow">ONE SHARED NETWORK · TWO EXECUTION PATHS</span>
            <h1>先算当前 token，再看历史怎样复用。</h1>
            <p>上方是不区分缓存策略的共同计算网络；下方才比较：无缓存需要重算什么，有缓存需要新算什么、已经保存了什么。</p>
          </div>
          <div className="baseline"><b>读图顺序</b><br />① 当前 Q／K／V → ② 两条路径<br />③ 注意力算子与数值展开</div>
        </section>

        <aside className={`replay-dock ${dockOpen ? "" : "is-collapsed"}`} id="replayDock" aria-label="悬浮区域导航、上下文与步骤控制">
          <nav className="section-dock" id="sectionDock" aria-label="页面区域导航">
            <strong>滚动到</strong>
            {[
              ["shared-computation", "①", "注意力模块"],
              ["execution-comparison", "②", "双轨对照"],
              ["numeric-expansion", "③", "数值展开"],
            ].map(([id, number, label]) => (
              <a href={`#${id}`} className={activeSection === id ? "is-active" : ""} onClick={(event) => { event.preventDefault(); scrollToSection(id); }} key={id}>
                <b>{number}</b><span>{label}</span>
              </a>
            ))}
          </nav>
          <button className="dock-toggle" id="dockToggle" type="button" aria-expanded={dockOpen} aria-controls="dockBody" onClick={() => setDockOpen((open) => !open)}>
            <span><b>上下文与步骤</b><small id="dockStatus">{prefill ? "PREFILL" : `DECODE ${step}`} · {t} tokens</small></span>
            <i aria-hidden="true">{dockOpen ? "−" : "+"}</i>
          </button>
          <div className="dock-body" id="dockBody">
            <div className="transport-bar">
              <div className="phase-readout">
                <span id="phase">{prefill ? "PREFILL" : `DECODE ${step}`}</span>
                <strong id="narration">{prefill ? `并行计算 ${t} 个提示 token，建立 K／V 缓存` : `生成“${ALL[current]}” · 上下文 ${t} tokens`}</strong>
              </div>
              <div className="transport">
                <button className="button" id="restart" type="button" onClick={() => stopAndGo(0)}>回到 Prefill</button>
                <button className="button" id="prev" type="button" disabled={step === 0} onClick={() => stopAndGo(step - 1)}>上一步</button>
                <button className="button primary" id="play" type="button" onClick={() => {
                  if (playing) {
                    setPlaying(false);
                  } else {
                    if (step === LAST_STEP) goToStep(0);
                    setPlaying(true);
                  }
                }}>{playing ? "暂停" : step === LAST_STEP ? "重新播放" : "播放"}</button>
                <button className="button" id="next" type="button" disabled={step === LAST_STEP} onClick={() => stopAndGo(step + 1)}>下一步</button>
              </div>
            </div>
            <div className="token-strip">
              <span>当前上下文</span>
              <p className="context-sentence" id="contextSentence">{ALL.slice(0, t).join("")}</p>
              <div className="token-list" id="tokens" aria-label="token 序列">
                {ALL.map((token, index) => <span className={`token ${index === current ? "current" : index < t ? "old" : "future"}`} key={index}>{token}</span>)}
              </div>
              <small id="tokenCount" className="mono">{t} / 9 positions</small>
            </div>
            <div className="timeline">
              <label htmlFor="scrub">生成时间轴</label>
              <input id="scrub" type="range" min="0" max={LAST_STEP} value={step} step="1" onChange={(event) => stopAndGo(Number(event.target.value))} />
              <output id="progress">{step} / {LAST_STEP}</output>
            </div>
          </div>
        </aside>

        <section className="replay-console" aria-label="KV Cache 共享网络与双轨对照">
          <section className="section-block" id="shared-computation">
            <div className="section-heading">
              <div><span>① SHARED COMPUTATION / 不区分有无缓存</span><h2>当前 token 在一个 Head 中怎样计算</h2><p>浅蓝高亮表示本阶段实际执行的算子；黄色描边表示正在查看的矩阵。点击 QKᵀ、Mask、Softmax 或 AV，可切换下方矩阵。</p></div>
              <code>HEAD 0 · dhead = 4</code>
            </div>
            <div className="shared-layout">
              <div className="network-panel">
                <div className="network-title"><strong>Transformer Block · 注意力计算网络</strong><code>SHARED</code></div>
                <div className="network-scroll">
                  <svg className="network-svg" id="sharedNetwork" viewBox="0 0 720 730" role="img" aria-label="当前 token 的共享注意力计算网络">
                    <defs><marker id="kv-arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="oklch(76% .018 260)" /></marker></defs>
                    <text className="phase-label" x="20" y="24">MULTI-HEAD ATTENTION / HEAD 0</text>
                    <path className="edge" d="M360 67V85" /><text className="trunk-label" x="360" y="55" textAnchor="middle">X · Block 输入</text>
                    <g className="node"><rect x="270" y="85" width="180" height="58" rx="3" /><text x="360" y="111">RMSNorm</text><text className="sub" x="360" y="129">X̂ · 注意力前归一化</text></g>
                    <path className="edge q-edge" d="M340 143V170H140V194" /><path className="edge k-edge" d="M360 143V194" /><path className="edge v-edge" d="M380 143V170H580V194" />
                    <g className="node q is-active" data-projection="q"><rect x="55" y="194" width="170" height="66" rx="3" /><text x="140" y="222">WQ,0</text><text className="sub" x="140" y="243">Q₀ · 当前提问</text></g>
                    <g className="node k is-active" data-projection="k"><rect x="275" y="194" width="170" height="66" rx="3" /><text x="360" y="222">WK,0</text><text className="sub" x="360" y="243">K₀ · 匹配索引</text></g>
                    <g className="node v is-active" data-projection="v"><rect x="495" y="194" width="170" height="66" rx="3" /><text x="580" y="222">WV,0</text><text className="sub" x="580" y="243">V₀ · 内容表示</text></g>
                    <g><circle className="step-badge" cx="211" cy="208" r="13" /><text className="step-text" x="211" y="212">1</text><circle className="step-badge" cx="431" cy="208" r="13" /><text className="step-text" x="431" y="212">2</text><circle className="step-badge" cx="651" cy="208" r="13" /><text className="step-text" x="651" y="212">3</text></g>
                    <path className="edge q-edge" d="M140 260V286H325V310" /><path className="edge k-edge" d="M360 260V310" />
                    <NetworkOperator focus="scores" currentFocus={focus} label="QKᵀ / √d" sub="④ QK 点积与缩放" x={250} y={310} onSelect={selectOperator} />
                    <path className="edge" d="M360 372V402" />
                    <NetworkOperator focus="mask" currentFocus={focus} label="Causal mask" sub="⑤ 遮蔽未来 token" x={250} y={402} onSelect={selectOperator} />
                    <path className="edge" d="M360 464V494" />
                    <NetworkOperator focus="softmax" currentFocus={focus} label="Softmax" sub="⑥ 每行权重归一化" x={250} y={494} onSelect={selectOperator} />
                    <path className="edge" d="M360 556V586" /><path className="edge v-edge" d="M580 260V608H470" /><text className="trunk-label" x="585" y="586">V₀ · 旁路</text>
                    <NetworkOperator focus="av" currentFocus={focus} label="AV" sub="⑦ 注意力权重乘 V" x={250} y={586} onSelect={selectOperator} />
                    <path className="edge" d="M360 648V676" />
                    <g className="node is-active" data-output><rect x="225" y="676" width="270" height="42" rx="3" /><text x="360" y="702">Concat → ZW_O → X + P</text></g>
                  </svg>
                </div>
              </div>

              <aside className="qkv-panel" id="qkvPanel" aria-label="当前阶段参与计算的 Q K V 操作数">
                <div className="qkv-head">
                  <strong>{prefill ? "PREFILL / 一次完成整段提示词的注意力计算" : "DECODE · 当前 token 的实际操作数"}</strong>
                  <span>{prefill ? `并行计算 ${t} 个提示 token：Q／K／V 均为 [${t},4]，一次完成全部因果注意力并建立缓存。` : `当前 Q 为 [1,4]；本次注意力读取 K／V 各 ${t} 行（position 0…${current}）。`}</span>
                </div>
                <section className="qkv-item q" data-operand="q">
                  <div className="qkv-item-title"><strong>1. Q₀ · {prefill ? "全部提示 Query" : "当前 Query"}</strong><code>Q[{prefill ? `0…${current}` : current}] = X̂[{prefill ? `0…${current}` : current}]WQ,0</code></div>
                  <p>{prefill ? `Q₀…Q${current} 全部并行产生，共 ${t} 行；下面完整展示每一行，所有行都参与因果注意力。` : `只使用当前 Q${current} 这一行，分别与下面 ${t} 个 K 做点积。`}</p>
                  {prefill ? <><div className="operand-range">行 = Query token · 列 = 4 维向量</div><OperandRows kind="q" data={Q} t={t} prefill={prefill} current={current} /></> : <div className="vector">{Q[current].map((value, d) => <i key={d}>{fmt(value)}</i>)}</div>}
                  <div className="vector-caption">[{prefill ? t : 1},4] · {prefill ? "完整提示 Query 矩阵" : "仅当前 Query 行"}</div>
                </section>
                <section className="qkv-item k" data-operand="k">
                  <div className="qkv-item-title"><strong>2. K₀ · {prefill ? "全部提示 Key" : "参与匹配的全部 Key"}</strong><code>K[0…{current}] = X̂[0…{current}]WK,0</code></div>
                  <p>{prefill ? `K₀…K${current} 全部并行产生。第 q 行只读取 K₀…K_q，未来位置由因果 Mask 遮蔽。` : `当前 Q 要匹配 K₀…K${current}，共 ${t} 行；不是只有当前 token 的 K${current}。`}</p>
                  <div className="operand-range">行 = Key token · 列 = 4 维向量</div>
                  <OperandRows kind="k" data={K} t={t} prefill={prefill} current={current} />
                  <div className="vector-caption">[{t},4] · {prefill ? "完整提示 Key 矩阵" : `当前行 Q${current} 的全部 Key 操作数`}</div>
                </section>
                <section className="qkv-item v" data-operand="v">
                  <div className="qkv-item-title"><strong>3. V₀ · {prefill ? "全部提示 Value" : "参与加权的全部 Value"}</strong><code>V[0…{current}] = X̂[0…{current}]WV,0</code></div>
                  <p>{prefill ? `V₀…V${current} 全部并行产生。每个 Q 行用自己的注意力权重，对可见 V 行加权求和。` : `Softmax 产生 ${t} 个权重后，对 V₀…V${current} 加权求和；不是只使用当前 V${current}。`}</p>
                  <div className="operand-range">行 = Value token · 列 = 4 维向量</div>
                  <OperandRows kind="v" data={V} t={t} prefill={prefill} current={current} />
                  <div className="vector-caption">[{t},4] · {prefill ? "完整提示 Value 矩阵" : `当前行 Q${current} 的全部 Value 操作数`}</div>
                </section>
              </aside>
            </div>
            <div className="shared-note" id="sharedNote">
              {prefill
                ? <><strong>Prefill · 一次处理完整提示词：</strong>并行产生 {t} 个提示 token 的 Q／K／V，计算全部 {t} 个 Query 行的因果注意力；右侧建立 {t} 份 K／V 缓存，随后用最后一行“{ALL[current]}”的表示预测下一 token。</>
                : <><strong>Decode：</strong>只展开当前“{ALL[current]}”的 1 行 Q，以及实际读取的 {t} 行 K／V。无缓存重算全部 K／V；有缓存只追加当前项，历史 {t - 1} 行直接读取。</>}
            </div>
          </section>

          <div className="duel" id="execution-comparison">{renderLane("uncached")}{renderLane("cached")}</div>

          <section className="cell-inspector" id="numeric-expansion" aria-live="polite">
            <div className="inspector-head"><div><span>③ NUMERIC EXPANSION / 两轨数值相同</span><h3 id="inspectorTitle">{inspector.title}</h3></div><output id="inspectorValue">{fmt(inspectorValue)}</output></div>
            <code className="numeric-expression" id="inspectorExpression">{inspector.expression}</code>
            <div className="term-grid" id="inspectorTerms">{inspector.terms.map((term, index) => <span key={index}>{term}</span>)}</div>
            <p id="inspectorNote">{inspector.note}</p>
          </section>
          <div className="comparison-note" id="comparisonNote">
            {prefill
              ? <><strong>Prefill 对照：</strong>两边都并行计算完整提示词的所有 Q 行。右侧把 K／V 保存下来；左侧没有可跨轮使用的缓存。</>
              : <><strong>Decode 对照：</strong>两边都只算当前 1 行 Q，并完成相同的 {t} 个 Q→K 连接。左侧因没有缓存，需要重算全部 {t} 份 K／V；右侧只新算当前 1 份，直接读取已有 {t - 1} 份。历史 Q 与历史 QK 都不是当前输出的必需计算。</>}
          </div>
        </section>

        <section className="lesson">
          <div><span className="eyebrow">WHY K／V, NOT Q?</span><h2>为什么只存 K／V，<br />不把 Q 也存起来？</h2></div>
          <div className="lesson-body">
            <div className="qkv-grid">
              <section><h3>Q · 提问</h3><p>历史 Q 已经完成自己的注意力计算。生成新 token 时只需要当前 Q，所以历史 Q 通常无需缓存。</p></section>
              <section><h3>K · 索引</h3><p>当前 Q 与所有历史 K 做点积，得到 S = QKᵀ／√d。历史 K 会在之后每轮反复使用。</p></section>
              <section><h3>V · 内容</h3><p>Softmax 权重对历史 V 加权求和，得到 C = AV。历史 V 同样会反复使用。</p></section>
            </div>
            <p className="fine">也不存完整注意力矩阵：S／A 可由当前 Q 与缓存 K 重新计算，完整矩阵会随上下文平方增长；K／V 缓存只随上下文线性增长。</p>
          </div>
        </section>

        <section className="lesson">
          <div><span className="eyebrow">MEMORY / LINEAR GROWTH</span><h2>这份“记忆”<br />到底占多少空间？</h2></div>
          <div className="lesson-body cache-calculator">
            <div className="cache-tool">
              <p>对常规 MHA／GQA，每个 token 要在<b>每一层</b>保存 K 和 V。示例固定为 32 层、8 个 KV Head、每 Head 128 维、BF16（2 字节）。</p>
              <label htmlFor="context"><span>上下文长度 T <strong><b id="tLabel">{contextLength.toLocaleString()}</b> tokens</strong></span><input id="context" type="range" min="128" max="32768" step="128" value={contextLength} onChange={(event) => setContextLength(Number(event.target.value))} /></label>
              <div className="cache-result"><span>每条序列、全部层的 KV Cache</span><strong id="mem">{cacheBytes >= 1073741824 ? `${(cacheBytes / 1073741824).toFixed(2)} GiB` : `${(cacheBytes / 1048576).toFixed(0)} MiB`}</strong><small id="memDetail">{cacheBytes.toLocaleString()} 字节 / 序列 · 每 token 128 KiB</small></div>
            </div>
            <div><div className="formula">Cache = B × T × L × 2(K,V) × Hkv × dhead × 字节数<br /><br />本例 B=1：每 token = 32 × 2 × 8 × 128 × 2 B = 128 KiB</div><p className="fine">上下文翻倍，缓存翻倍；Batch size 翻倍，缓存也翻倍。这里不含模型权重、激活、Logits、工作区与框架额外开销。</p></div>
          </div>
        </section>
        <div className="takeaway">一句话记住：KV Cache 把“过去已经算过的 K／V”留在显存里，<b>用空间换时间</b>；它不会消除当前 Q 对历史的读取。</div>
      </main>
      <footer className="site-footer"><span>INFERENCE LAB</span><p>KV Cache 生成回放 · 本地确定性数值 · 不读取或上传任何数据</p></footer>
    </div>
  );
}
