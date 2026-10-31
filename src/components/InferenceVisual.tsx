import { useMemo, useState } from "react";
import type { Candidate, ForwardTrace, Matrix, Token } from "@/lib/tinyTransformer";
import { MODEL_CONFIG, VOCAB } from "@/lib/tinyTransformer";

const BLUE = [32, 82, 187];
const RED = [229, 67, 46];

function cellColor(value: number, max = 1) {
  const strength = Math.min(Math.abs(value) / Math.max(max, 0.001), 1);
  const color = value >= 0 ? BLUE : RED;
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${0.08 + strength * 0.82})`;
}

export function VectorStrip({ vector, label }: { vector: number[]; label?: string }) {
  const max = Math.max(...vector.map(Math.abs), 0.001);
  return (
    <div className="vector-wrap" aria-label={label ?? "向量可视化"}>
      {label && <span className="micro-label">{label}</span>}
      <div className="vector-strip">
        {vector.map((value, index) => (
          <div className="vector-cell" key={index} title={`维度 ${index}: ${value.toFixed(4)}`}>
            <div className="vector-color" style={{ backgroundColor: cellColor(value, max) }} />
            <span>{value.toFixed(2)}</span>
            <small>d{index}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

export function MatrixHeatmap({
  matrix,
  tokens,
  selectedToken,
  onSelectToken,
  label,
}: {
  matrix: Matrix;
  tokens: Token[];
  selectedToken: number;
  onSelectToken?: (index: number) => void;
  label: string;
}) {
  const max = Math.max(...matrix.flat().map(Math.abs), 0.001);
  return (
    <div className="heatmap-scroll" tabIndex={0} aria-label={`${label}，可横向滚动`}>
      <div
        className="matrix-heatmap"
        style={{ gridTemplateColumns: `minmax(62px, 1.4fr) repeat(${matrix[0].length}, minmax(36px, 1fr))` }}
      >
        <div className="matrix-corner">token \ d</div>
        {matrix[0].map((_, dimension) => (
          <div className="matrix-heading" key={`h-${dimension}`}>d{dimension}</div>
        ))}
        {matrix.map((row, tokenIndex) => (
          <div className="matrix-row" key={`${tokens[tokenIndex]?.text}-${tokenIndex}`} style={{ display: "contents" }}>
            <button
              type="button"
              className={`matrix-token ${selectedToken === tokenIndex ? "is-selected" : ""}`}
              onClick={() => onSelectToken?.(tokenIndex)}
              aria-label={`查看位置 ${tokenIndex} 的 ${tokens[tokenIndex]?.text}`}
            >
              <span>{tokens[tokenIndex]?.text}</span><small>{tokenIndex}</small>
            </button>
            {row.map((value, dimension) => (
              <div
                className={`matrix-value ${selectedToken === tokenIndex ? "is-selected" : ""}`}
                key={`${tokenIndex}-${dimension}`}
                style={{ backgroundColor: cellColor(value, max) }}
                title={`${tokens[tokenIndex]?.text} · d${dimension} = ${value.toFixed(4)}`}
              >
                {value.toFixed(2)}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="heatmap-legend"><span>负值</span><i className="legend-negative" /><i className="legend-neutral" /><i className="legend-positive" /><span>正值</span></div>
    </div>
  );
}

export function AttentionHeatmap({
  trace,
  layer,
  head,
  query,
  onQueryChange,
}: {
  trace: ForwardTrace;
  layer: number;
  head: number;
  query: number;
  onQueryChange: (index: number) => void;
}) {
  const [activeCell, setActiveCell] = useState<{ row: number; col: number } | null>(null);
  const weights = trace.layers[layer].attention[head];
  const selected = activeCell ?? { row: query, col: Math.min(query, weights.length - 1) };
  return (
    <div className="attention-layout">
      <div className="attention-scroll" tabIndex={0} aria-label="注意力矩阵，可横向滚动">
        <div
          className="attention-grid"
          style={{ gridTemplateColumns: `minmax(64px, 1.25fr) repeat(${weights.length}, minmax(44px, 1fr))` }}
        >
          <div className="matrix-corner">Q \ K</div>
          {trace.tokens.map((token, index) => (
            <div className="matrix-heading token-heading" key={`k-${index}`}>{token.text}<small>K{index}</small></div>
          ))}
          {weights.map((row, i) => (
            <div key={`r-${i}`} style={{ display: "contents" }}>
              <button
                type="button"
                className={`matrix-token ${query === i ? "is-selected" : ""}`}
                onClick={() => onQueryChange(i)}
              >
                <span>{trace.tokens[i].text}</span><small>Q{i}</small>
              </button>
              {row.map((value, j) => (
                <button
                  type="button"
                  key={`${i}-${j}`}
                  className={`attention-cell ${j > i ? "is-masked" : ""} ${selected.row === i && selected.col === j ? "is-active" : ""}`}
                  style={{ backgroundColor: j > i ? undefined : cellColor(value, 1) }}
                  onClick={() => { setActiveCell({ row: i, col: j }); onQueryChange(i); }}
                  aria-label={j > i ? `位置 ${i} 到 ${j} 被因果掩码屏蔽` : `注意力 ${value.toFixed(4)}`}
                >
                  {j > i ? "×" : value.toFixed(2)}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="attention-readout" aria-live="polite">
        <span className="micro-label">当前注意力边</span>
        <strong>{trace.tokens[selected.row]?.text} <i>→</i> {trace.tokens[selected.col]?.text}</strong>
        {selected.col > selected.row ? (
          <p>未来位置被遮罩，softmax 后权重严格为 0。</p>
        ) : (
          <>
            <b>{(weights[selected.row][selected.col] * 100).toFixed(1)}%</b>
            <p>该 head 在计算位置 {selected.row} 时，从位置 {selected.col} 的 V 向量取回的信息比例。</p>
          </>
        )}
      </div>
    </div>
  );
}

export function ProbabilityChart({
  candidates,
  selectedId,
  onSelect,
}: {
  candidates: Candidate[];
  selectedId?: number;
  onSelect?: (candidate: Candidate) => void;
}) {
  return (
    <div className="probability-chart">
      {candidates.map((candidate, index) => (
        <button
          type="button"
          className={`probability-row ${selectedId === candidate.id ? "is-selected" : ""}`}
          key={candidate.id}
          onClick={() => onSelect?.(candidate)}
          disabled={!onSelect}
        >
          <span className="candidate-rank">{String(index + 1).padStart(2, "0")}</span>
          <span className="candidate-token">{candidate.token}</span>
          <span className="probability-track" aria-hidden="true">
            <i style={{ transform: `scaleX(${candidate.probability})` }} />
          </span>
          <strong>{(candidate.probability * 100).toFixed(1)}%</strong>
          <small>logit {candidate.logit.toFixed(2)}</small>
        </button>
      ))}
    </div>
  );
}

export function ActivationBars({ vector }: { vector: number[] }) {
  const max = Math.max(...vector.map(Math.abs), 0.001);
  return (
    <div className="activation-bars" aria-label="MLP 隐藏层激活值">
      {vector.map((value, index) => (
        <div className="activation-column" key={index} title={`隐藏神经元 ${index}: ${value.toFixed(4)}`}>
          <div className="activation-space">
            <i style={{ transform: `scaleY(${Math.abs(value) / max})`, backgroundColor: cellColor(value, max) }} />
          </div>
          <span>{index}</span>
        </div>
      ))}
    </div>
  );
}

export function ArchitectureStrip({ compact = false }: { compact?: boolean }) {
  const parts = useMemo(() => [
    ["Tokens", "n"], ["Embed", "8"], ["×2 Block", "8"], ["LM Head", String(VOCAB.length)], ["Next", "1"],
  ], []);
  return (
    <div className={`architecture-strip ${compact ? "is-compact" : ""}`} aria-label="模型张量形状">
      {parts.map(([name, shape], index) => (
        <div className="architecture-part" key={name}>
          <span>{name}</span><strong>{shape}</strong>
          {index < parts.length - 1 && <i aria-hidden="true">→</i>}
        </div>
      ))}
      {!compact && <small>n 个位置 · {MODEL_CONFIG.layers} 层 · {MODEL_CONFIG.heads} heads · d_model={MODEL_CONFIG.dModel}</small>}
    </div>
  );
}
