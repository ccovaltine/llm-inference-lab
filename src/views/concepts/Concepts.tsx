import { useMemo, useState } from "react";
import { ArrowDown, Database, Gauge, Layers3 } from "lucide-react";

export default function Concepts() {
  const [sequence, setSequence] = useState(4096);
  const [layers, setLayers] = useState(32);
  const cache = useMemo(() => {
    const dModel = 4096;
    const bytes = 2;
    const total = 2 * layers * sequence * dModel * bytes;
    return { total, gb: total / 1024 ** 3, perToken: total / sequence / 1024 };
  }, [layers, sequence]);

  return (
    <div className="concepts-page">
      <header className="view-intro">
        <span className="stage-kicker">MENTAL MODEL</span>
        <h1>把推理看成一条反复闭合的回路</h1>
        <p>训练负责“得到权重”，推理负责“使用权重”。推理时没有反向传播，也不会因为一次对话就自动更新模型参数。</p>
      </header>

      <section className="concept-section">
        <div className="concept-copy"><span className="concept-index">A</span><h2>一次生成，两种计算节奏</h2><p>Prefill 并行处理整段 prompt，Decode 串行生成新 token。前者更像矩阵的大批量计算，后者更容易受每一步读取历史状态的开销影响。</p></div>
        <div className="prefill-diagram">
          <div className="prefill-row"><span>Prefill</span><div>{["今", "天", "天", "气"].map((token, i) => <b key={i}>{token}</b>)}</div><strong>并行 → 第 1 个新 token</strong></div>
          <ArrowDown />
          <div className="decode-row"><span>Decode</span><div>{["很", "好", "。", "<EOS>"].map((token, i) => <b key={i}>{token}<small>step {i + 1}</small></b>)}</div><strong>逐步串行</strong></div>
        </div>
      </section>

      <section className="concept-section cache-calculator">
        <div className="concept-copy"><span className="concept-index">B</span><h2>KV Cache 用显存换重复计算</h2><p>每一层都要保存每个历史 token 的 K 和 V。以一个 32 层、d_model=4096、FP16 的模型为例，调整上下文长度观察缓存规模。</p></div>
        <div className="cache-tool">
          <label><span>上下文长度 <strong>{sequence.toLocaleString()} tokens</strong></span><input type="range" min="512" max="32768" step="512" value={sequence} onChange={(event) => setSequence(Number(event.target.value))} /></label>
          <label><span>层数 <strong>{layers}</strong></span><input type="range" min="2" max="80" step="2" value={layers} onChange={(event) => setLayers(Number(event.target.value))} /></label>
          <div className="cache-result"><Database /><span>估算 KV Cache</span><strong>{cache.gb.toFixed(2)} GiB</strong><small>每 token 约 {cache.perToken.toFixed(0)} KiB · 仅含 K/V，不含模型权重和激活</small></div>
        </div>
        <div className="formula">bytes = 2（K,V）× layers × sequence × d_model × dtype_bytes</div>
      </section>

      <section className="concept-section">
        <div className="concept-copy"><span className="concept-index">C</span><h2>三个最容易混淆的地方</h2></div>
        <div className="misconception-list">
          <article><Gauge /><div><h3>Softmax 概率 ≠ 模型对事实的把握</h3><p>它只是在当前词表和上下文中，对“下一个 token”的相对偏好。高概率不等于内容真实。</p></div></article>
          <article><Layers3 /><div><h3>Attention 权重 ≠ 完整的因果解释</h3><p>它展示信息混合的一种方式，但模型还包含 MLP、残差和后续层，不能仅凭一张热图解释最终行为。</p></div></article>
          <article><Database /><div><h3>上下文里的内容 ≠ 已写进模型权重</h3><p>普通推理只把对话放在临时上下文与缓存中。持久记忆需要外部存储，改变权重则需要训练或微调。</p></div></article>
        </div>
      </section>
    </div>
  );
}
