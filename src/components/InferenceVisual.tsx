import { useMemo } from "react";
import type { Candidate } from "@/lib/tinyTransformer";
import { MODEL_CONFIG, VOCAB } from "@/lib/tinyTransformer";




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
