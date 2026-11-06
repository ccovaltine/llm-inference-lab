import { useOutletContext } from "react-router";
import { ArrowRight, LoaderCircle, Play, Save } from "lucide-react";
import { IS_STATIC_DEMO } from "@/const";
import TransformerWorkbench from "@/components/TransformerWorkbench";
import { ArchitectureStrip } from "@/components/InferenceVisual";
import { MODEL_CONFIG } from "@/lib/tinyTransformer";
import type { HomeOutletContext } from "@/pages/Home";

export default function LabView() {
  const {
    prompt,
    trace,
    candidates,
    selectedCandidate,
    generatedCount,
    canGenerate,
    temperature,
    topK,
    labRef,
    examples,
    draft,
    onDraftChange,
    inputError,
    onRunPrompt,
    onTemperatureChange,
    onTopKChange,
    onSelectCandidate,
    onCommitCandidate,
    isAuthenticated,
    createPending,
    onSaveCurrent,
    onSwitchView,
  } = useOutletContext<HomeOutletContext>();
  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <span className="hero-kicker">AN END-TO-END TRANSFORMER, UNDER A MICROSCOPE</span>
          <h1>一个 token，<br />究竟怎样被算出来？</h1>
          <p>从一句话到下一个词：亲手走过分词、向量、两层 Transformer、Logits 与采样。这里不是流程动画，而是每一步都有真实数值的微型推理。</p>
          <button type="button" className="hero-cta" onClick={() => labRef.current?.scrollIntoView({ behavior: "smooth" })}>进入推理回路<ArrowRight /></button>
        </div>
        <div className="hero-model" aria-label="微型模型结构">
          <div className="hero-model-top"><span>LIVE MODEL / 2L-2H</span><i>可计算</i></div>
          <ArchitectureStrip />
          <div className="hero-equation"><span>h<sub>t</sub></span><b>→</b><span>softmax(W<sub>U</sub>h<sub>t</sub>)</span><b>→</b><strong>token<sub>t+1</sub></strong></div>
          <p>2 layers · 2 attention heads · d_model 8<br />固定权重，所有中间结果可检查</p>
        </div>
      </section>

      <section className="lab-section" id="lab" ref={labRef}>
        <div className="lab-toolbar">
          <div className="prompt-area">
            <label htmlFor="prompt">给微型模型一段上下文</label>
            <div className="prompt-input-row">
              <input id="prompt" value={draft} maxLength={80} onChange={(event) => onDraftChange(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") onRunPrompt(); }} aria-describedby={inputError ? "prompt-error" : undefined} />
              <button type="button" className="button primary" onClick={() => onRunPrompt()}><Play />运行推理</button>
            </div>
            {inputError && <p className="field-error" id="prompt-error">{inputError}</p>}
            <div className="example-row"><span>试试</span>{examples.map((example) => <button type="button" key={example} onClick={() => onRunPrompt(example)}>{example}</button>)}</div>
          </div>
          <div className="decode-controls">
            <label><span>Temperature <strong>{temperature.toFixed(1)}</strong></span><input type="range" min="0.1" max="2" step="0.1" value={temperature} onChange={(event) => onTemperatureChange(Number(event.target.value))} /></label>
            <label><span>Top-k <strong>{topK}</strong></span><input type="range" min="1" max="16" step="1" value={topK} onChange={(event) => onTopKChange(Number(event.target.value))} /></label>
            {!IS_STATIC_DEMO && <button type="button" className="button secondary save-button" onClick={isAuthenticated ? onSaveCurrent : () => onSwitchView("notebook")} disabled={createPending}>{createPending ? <LoaderCircle className="spin" /> : <Save />}{isAuthenticated ? "保存实验" : "登录后保存"}</button>}
          </div>
        </div>

        <div className="generation-status" aria-live="polite">
          <span className="status-label">当前上下文</span>
          <div className="status-tokens">{trace.tokens.map((token, index) => <span className={index >= trace.tokens.length - generatedCount && generatedCount > 0 ? "is-generated" : ""} key={`${token.text}-${index}`}>{token.text}</span>)}</div>
          <small>{trace.tokens.length}/{MODEL_CONFIG.maxTokens} positions · 已生成 {generatedCount} tokens</small>
        </div>

        <TransformerWorkbench
          key={prompt}
          trace={trace}
          candidates={candidates}
          selectedCandidate={selectedCandidate}
          onSelectCandidate={onSelectCandidate}
          onCommitCandidate={onCommitCandidate}
          generatedCount={generatedCount}
          canGenerate={canGenerate}
          temperature={temperature}
          topK={topK}
        />
      </section>
      <footer className="site-footer"><span>INFERENCE LAB</span><p>教学模型的输出不代表真实语言能力；它的价值是让每个算子、形状和状态变化都可见。</p><button type="button" onClick={() => onSwitchView("concepts")}>继续理解 KV Cache<ArrowRight /></button></footer>
    </main>
  );
}
