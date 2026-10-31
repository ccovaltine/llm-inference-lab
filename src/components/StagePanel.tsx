import { ArrowRight, Braces, Eye, MousePointer2 } from "lucide-react";
import type { Candidate, ForwardTrace } from "@/lib/tinyTransformer";
import { MODEL_CONFIG, VOCAB, getCandidates, tokensToText } from "@/lib/tinyTransformer";
import {
  ActivationBars,
  ArchitectureStrip,
  AttentionHeatmap,
  MatrixHeatmap,
  ProbabilityChart,
  VectorStrip,
} from "@/components/InferenceVisual";

export const STAGES = [
  { name: "请求进入", short: "输入", phase: "Prefill" },
  { name: "分词与 ID", short: "Tokenize", phase: "Prefill" },
  { name: "向量与位置", short: "Embed", phase: "Prefill" },
  { name: "第 1 层注意力", short: "Attn 1", phase: "Prefill" },
  { name: "MLP 与残差", short: "MLP 1", phase: "Prefill" },
  { name: "第 2 层重复", short: "Block 2", phase: "Prefill" },
  { name: "归一化与 LM Head", short: "Logits", phase: "Decode" },
  { name: "Softmax 与采样", short: "Sample", phase: "Decode" },
  { name: "回填，进入下一轮", short: "Append", phase: "Loop" },
] as const;

type Props = {
  trace: ForwardTrace;
  stage: number;
  selectedToken: number;
  onSelectToken: (index: number) => void;
  head: number;
  onHeadChange: (head: number) => void;
  candidates: Candidate[];
  selectedCandidate: Candidate;
  onSelectCandidate: (candidate: Candidate) => void;
  onCommitCandidate: (candidate: Candidate) => void;
  onRandomCandidate: () => void;
  onNextCycle: () => void;
  generatedCount: number;
  canGenerate: boolean;
};

function Formula({ children }: { children: React.ReactNode }) {
  return <div className="formula">{children}</div>;
}

function Note({ title, children }: { title: string; children: React.ReactNode }) {
  return <aside className="stage-note"><strong>{title}</strong><p>{children}</p></aside>;
}

export default function StagePanel(props: Props) {
  const {
    trace, stage, selectedToken, onSelectToken, head, onHeadChange,
    candidates, selectedCandidate, onSelectCandidate, onCommitCandidate,
    onRandomCandidate, onNextCycle, generatedCount, canGenerate,
  } = props;
  const tokenIndex = Math.min(selectedToken, trace.tokens.length - 1);
  const token = trace.tokens[tokenIndex];
  const lastLayer = trace.layers[trace.layers.length - 1];

  if (stage === 0) {
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">PREFILL / 01</span><h2>模型收到的不是“一句话”，而是一次前向计算任务</h2><p>服务端验证请求后，把文本、温度等解码参数交给模型。权重在推理前已经加载；生成过程中不会更新权重。</p></div>
        <div className="request-flow" aria-label="推理请求流程">
          {["你的文本", "Tokenizer", "Transformer 权重", "下一个 token"].map((item, index) => (
            <div className="request-node" key={item}><span>{item}</span>{index < 3 && <ArrowRight aria-hidden="true" />}</div>
          ))}
        </div>
        <div className="two-col-explain">
          <div><span className="micro-label">预填充 Prefill</span><h3>一次看完已有上下文</h3><p>对 prompt 的全部位置并行计算，建立每一层的 K、V。此后才能得到第一个新 token。</p></div>
          <div><span className="micro-label">解码 Decode</span><h3>每次只新增一个 token</h3><p>把新 token 接回上下文，再前向一次。真实系统会复用 KV Cache，而不是重复计算整段 prompt。</p></div>
        </div>
        <Note title="本页的计算口径">这是一个 {MODEL_CONFIG.layers} 层、{MODEL_CONFIG.heads} 头、d_model={MODEL_CONFIG.dModel} 的 Decoder-only Transformer。所有矩阵都真实计算；固定伪随机权重仅用于教学，不具备语言智能。</Note>
      </div>
    );
  }

  if (stage === 1) {
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">TOKENIZE / 02</span><h2>文本被切成词表中的整数 ID</h2><p>Embedding 层只能查表，不能直接理解字符。这里用最长匹配的小词表模拟 BPE／SentencePiece 的角色。</p></div>
        <div className="token-lane">
          {trace.tokens.map((item, index) => (
            <button type="button" key={`${item.text}-${index}`} className={`token-chip ${item.unknown ? "is-unknown" : ""} ${tokenIndex === index ? "is-selected" : ""}`} onClick={() => onSelectToken(index)}>
              <small>#{index}</small><strong>{item.text}</strong><span>ID {item.id}</span>
            </button>
          ))}
        </div>
        <Formula>“{tokensToText(trace.tokens)}” → [{trace.tokens.map((item) => item.id).join(", ")}]</Formula>
        <div className="two-col-explain">
          <div><h3>同一个 token，同一个 ID</h3><p>ID 只是词表索引，不包含语义大小关系。ID 10 并不比 ID 3 “更大”或“更重要”。</p></div>
          <div><h3>未知文本不会被悄悄丢掉</h3><p>超出教学词表的内容显示为红色，并映射到 &lt;UNK&gt;。真实模型通常用子词拆分来避免 Unknown。</p></div>
        </div>
      </div>
    );
  }

  if (stage === 2) {
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">EMBED / 03</span><h2>查词向量，再加上“我在第几个位置”</h2><p>每个 ID 从 Embedding 矩阵取出 8 个数。位置向量让模型区分“猫追狗”和“狗追猫”。</p></div>
        <MatrixHeatmap matrix={trace.embeddings} tokens={trace.tokens} selectedToken={tokenIndex} onSelectToken={onSelectToken} label="Token 加位置 Embedding" />
        <div className="vector-equation">
          <VectorStrip vector={trace.tokenEmbeddings[tokenIndex]} label={`${token.text} · token embedding`} />
          <span className="math-operator">+</span>
          <VectorStrip vector={trace.positionEmbeddings[tokenIndex]} label={`位置 ${tokenIndex} · position embedding`} />
          <span className="math-operator">=</span>
          <VectorStrip vector={trace.embeddings[tokenIndex]} label="输入到 Block 1 的 hidden state" />
        </div>
        <Formula>X<sub>0</sub> = E<sub>token</sub>[token_id] + E<sub>position</sub>[position]　→　形状 [{trace.tokens.length}, {MODEL_CONFIG.dModel}]</Formula>
      </div>
    );
  }

  if (stage === 3) {
    const layer = trace.layers[0];
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">BLOCK 1 / ATTENTION</span><h2>每个位置只向“过去”收集信息</h2><p>Q 决定我要找什么，K 决定我有什么，V 是真正被取回的内容。点击矩阵单元格，追踪一条注意力边。</p></div>
        <div className="segmented" aria-label="选择注意力头">
          {[0, 1].map((item) => <button type="button" key={item} className={head === item ? "is-active" : ""} onClick={() => onHeadChange(item)}>Head {item + 1}<small>4 维</small></button>)}
        </div>
        <AttentionHeatmap trace={trace} layer={0} head={head} query={tokenIndex} onQueryChange={onSelectToken} />
        <Formula>A = softmax((QK<sup>T</sup>) / √4 + causal_mask) · V</Formula>
        <div className="qkv-grid">
          <VectorStrip vector={layer.queries[head][tokenIndex]} label={`Q · ${token.text}`} />
          <VectorStrip vector={layer.keys[head][Math.min(tokenIndex, layer.keys[head].length - 1)]} label={`K · 同位置`} />
          <VectorStrip vector={layer.values[head][tokenIndex]} label={`V · 同位置`} />
        </div>
        <Note title="为什么上三角全是 ×？">生成第 i 个位置时，未来 token 尚不存在。掩码把 i 之后的 score 设为 −∞，softmax 后权重变成 0，防止模型“偷看答案”。</Note>
      </div>
    );
  }

  if (stage === 4) {
    const layer = trace.layers[0];
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">BLOCK 1 / MLP</span><h2>注意力混合信息，MLP 再逐位置变换信息</h2><p>MLP 不跨 token 交流。它把每个 8 维向量扩展到 16 维，经 GELU 非线性筛选，再投影回 8 维。</p></div>
        <div className="mlp-shape-flow"><span>8 维 hidden</span><ArrowRight /><span>16 维 GELU</span><ArrowRight /><span>8 维输出</span></div>
        <ActivationBars vector={layer.mlpHidden[tokenIndex]} />
        <div className="selected-readout"><MousePointer2 size={16} /><span>正在查看</span><strong>{token.text}</strong><small>位置 {tokenIndex} · 点击上方矩阵或 token 可切换</small></div>
        <MatrixHeatmap matrix={layer.output} tokens={trace.tokens} selectedToken={tokenIndex} onSelectToken={onSelectToken} label="第一个 Block 输出" />
        <Formula>X′ = X + Attention(RMSNorm(X))<br />X<sub>1</sub> = X′ + W₂ · GELU(W₁ · RMSNorm(X′) + b₁) + b₂</Formula>
        <Note title="残差连接不是可选项的装饰">“原始输入 + 新计算结果”让信息有直通路径，也使深层网络更容易训练。RMSNorm 则把数值尺度稳定下来。</Note>
      </div>
    );
  }

  if (stage === 5) {
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">BLOCK 2 / REPEAT</span><h2>同样的结构再来一次，但使用另一套权重</h2><p>第二层读取第一层的输出，而不是原始 Embedding。真实模型会把这个 Block 重复几十到上百次。</p></div>
        <div className="segmented" aria-label="选择第二层注意力头">
          {[0, 1].map((item) => <button type="button" key={item} className={head === item ? "is-active" : ""} onClick={() => onHeadChange(item)}>Layer 2 · Head {item + 1}<small>独立 Q/K/V</small></button>)}
        </div>
        <AttentionHeatmap trace={trace} layer={1} head={head} query={tokenIndex} onQueryChange={onSelectToken} />
        <details className="tensor-details">
          <summary><Braces size={17} />展开第二层的最终 hidden states</summary>
          <MatrixHeatmap matrix={lastLayer.output} tokens={trace.tokens} selectedToken={tokenIndex} onSelectToken={onSelectToken} label="第二层输出" />
        </details>
        <div className="two-col-explain">
          <div><h3>层与层不共享权重</h3><p>结构相同不代表参数相同。每层都有自己的 Q/K/V/O 和 MLP 矩阵。</p></div>
          <div><h3>所有位置仍保留独立向量</h3><p>最终张量仍是 [{trace.tokens.length}, 8]。预测下一个 token 时，只取最后一个位置的 hidden state。</p></div>
        </div>
      </div>
    );
  }

  if (stage === 6) {
    const rawCandidates = getCandidates(trace.logits, 1, VOCAB.length).slice(0, 8);
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">LM HEAD / 07</span><h2>把最后一个 hidden state 翻译成整份词表的分数</h2><p>先 RMSNorm，再乘以 Unembedding 矩阵。每个词得到一个 logit：分数尚未是概率，也没有被限制在 0 到 1。</p></div>
        <VectorStrip vector={trace.finalNormalized.at(-1)!} label={`最后位置 ${trace.tokens.at(-1)!.text} · final RMSNorm`} />
        <div className="logit-equation"><span>[1 × 8]</span><b>×</b><span>[8 × {VOCAB.length}]</span><b>+ bias</b><ArrowRight /><strong>[1 × {VOCAB.length}] logits</strong></div>
        <ProbabilityChart candidates={rawCandidates} />
        <Formula>logits = RMSNorm(h<sub>last</sub>) · W<sub>U</sub><sup>T</sup> + b</Formula>
        <Note title="为什么只取最后一个位置？">因果注意力已经让最后位置汇聚了前文信息，所以它是“下一个 token 应该是什么”的摘要。训练时会对所有位置并行算损失；自回归推理时通常只解码最后位置。</Note>
      </div>
    );
  }

  if (stage === 7) {
    return (
      <div className="stage-stack">
        <div className="stage-heading"><span className="stage-kicker">DECODE / 08</span><h2>概率分布不是答案；还要经过解码策略</h2><p>温度改变分布的尖锐程度，Top-k 先截断候选，再重新归一化。点击一个候选查看选择状态，或按概率随机抽一次。</p></div>
        <ProbabilityChart candidates={candidates} selectedId={selectedCandidate.id} onSelect={onSelectCandidate} />
        <div className="sampling-console">
          <div><span className="micro-label">当前选择</span><strong>{selectedCandidate.token}</strong><small>p = {(selectedCandidate.probability * 100).toFixed(2)}%</small></div>
          <div className="button-row">
            <button type="button" className="button secondary" onClick={onRandomCandidate} disabled={!canGenerate}>按概率抽一次</button>
            <button type="button" className="button primary" onClick={() => onCommitCandidate(selectedCandidate)} disabled={!canGenerate}>接回上下文<ArrowRight size={17} /></button>
          </div>
        </div>
        <Formula>p<sub>i</sub> = exp(logit<sub>i</sub> / T) / Σ<sub>j∈TopK</sub> exp(logit<sub>j</sub> / T)</Formula>
        {!canGenerate && <Note title="本轮已到停止条件">模型生成了 &lt;EOS&gt;，或已达到 {MODEL_CONFIG.maxTokens} 个位置。重置 prompt 后可以重新开始。</Note>}
      </div>
    );
  }

  return (
    <div className="stage-stack">
      <div className="stage-heading"><span className="stage-kicker">AUTOREGRESSIVE / 09</span><h2>一个 token 结束，也是下一次前向计算的起点</h2><p>新 token 被追加到上下文。下一轮会再次经过 Embedding 和所有 Block；已有 K/V 可以缓存，新的输出分布会随上下文改变。</p></div>
      <div className="autoregressive-loop">
        <div className="loop-context">
          {trace.tokens.map((item, index) => <span className={index === trace.tokens.length - 1 && generatedCount > 0 ? "is-new" : ""} key={`${item.text}-${index}`}>{item.text}{index === trace.tokens.length - 1 && generatedCount > 0 && <small>刚刚回填</small>}</span>)}
        </div>
        <div className="loop-arrow"><ArrowRight /><span>重新前向计算</span></div>
        <div className="loop-next"><Eye /><strong>新的 logits</strong><small>直到 EOS 或长度上限</small></div>
      </div>
      <ArchitectureStrip compact />
      <div className="cache-explainer">
        <div><span className="micro-label">没有 KV Cache</span><strong>重算整段上下文的 Q、K、V</strong><p>清晰但浪费。教学演示采用这种方式，方便展示完整矩阵。</p></div>
        <div><span className="micro-label">使用 KV Cache</span><strong>只为新 token 算 Q、K、V</strong><p>复用过去每层的 K/V；Attention 仍要读取全部历史，因此缓存会随上下文增长。</p></div>
      </div>
      <button type="button" className="button primary next-cycle" onClick={onNextCycle} disabled={!canGenerate}>从 Embedding 开始下一轮<ArrowRight size={17} /></button>
    </div>
  );
}
