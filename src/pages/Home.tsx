import { useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import {
  ArrowRight, BookOpen, CircleStop, Database, FlaskConical, FolderClock, LoaderCircle,
  LogIn, Network, Play, Save, Sparkles, Trash2, UserRound,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import "../App.css";
import { trpc } from "@/providers/trpc";
import { useAuth } from "@/hooks/useAuth";
import { IS_STATIC_DEMO } from "@/const";
import Concepts from "@/components/Concepts";
import TransformerWorkbench from "@/components/TransformerWorkbench";
import AttentionNetwork from "@/components/AttentionNetwork";
import KVCacheReplay from "@/components/KVCacheReplay";
import { ArchitectureStrip } from "@/components/InferenceVisual";
import {
  MODEL_CONFIG, forward, getCandidates, tokenize,
  type Candidate, type Token,
} from "@/lib/tinyTransformer";

const EXAMPLES = ["小猫坐在垫子上", "机器人站在沙发旁边", "小狗躺在草地上"];
type View = "lab" | "kv" | "attention" | "concepts" | "notebook";

export default function Home() {
  const [view, setView] = useState<View>("lab");
  const [draft, setDraft] = useState(EXAMPLES[0]);
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [generated, setGenerated] = useState<Token[]>([]);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(8);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const [inputError, setInputError] = useState("");
  const labRef = useRef<HTMLElement>(null);
  const { user, isAuthenticated, isLoading: authLoading, logout } = useAuth();
  const utils = trpc.useUtils();
  const experiments = trpc.experiment.list.useQuery(undefined, { enabled: isAuthenticated && !IS_STATIC_DEMO });
  const createExperiment = trpc.experiment.create.useMutation({
    onSuccess: async () => {
      await utils.experiment.list.invalidate();
      toast.success("实验已保存到你的记录");
    },
    onError: () => toast.error("保存失败，请稍后重试"),
  });
  const deleteExperiment = trpc.experiment.delete.useMutation({
    onSuccess: async () => {
      await utils.experiment.list.invalidate();
      toast.success("记录已删除");
    },
    onError: () => toast.error("删除失败，请重试"),
  });

  const trace = useMemo(() => forward([...tokenize(prompt), ...generated]), [prompt, generated]);
  const candidates = useMemo(
    () => getCandidates(trace.logits, temperature, topK),
    [trace.logits, temperature, topK],
  );
  const selectedCandidate = candidates.find((item) => item.id === selectedCandidateId) ?? candidates[0];
  const canGenerate = trace.tokens.length < MODEL_CONFIG.maxTokens && trace.tokens.at(-1)?.id !== 1;

  function runPrompt(nextPrompt = draft) {
    const clean = nextPrompt.trim();
    if (!clean) {
      setInputError("先输入一句话，模型才有上下文可以处理。");
      return;
    }
    if (clean.length > 60) {
      setInputError("教学演示最多输入 60 个字符，请缩短后再运行。");
      return;
    }
    setInputError("");
    setPrompt(clean);
    setDraft(clean);
    setGenerated([]);
    setSelectedCandidateId(null);
    toast.success("新的前向计算已就绪");
  }

  function commitCandidate(candidate: Candidate) {
    if (!canGenerate) return;
    const nextToken: Token = {
      id: candidate.id,
      text: candidate.token,
      raw: candidate.id === 1 ? "" : candidate.token,
      unknown: false,
    };
    setGenerated((current) => [...current, nextToken]);
    setSelectedCandidateId(null);
  }

  function switchView(next: View) {
    setView(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function saveCurrent() {
    if (!isAuthenticated) return;
    createExperiment.mutate({
      title: `推理实验 · ${prompt.slice(0, 14)}`,
      prompt,
      output: generated.length ? generated.map((token) => token.id === 1 ? "<EOS>" : token.raw).join(" ") : "尚未生成",
      temperature,
      topK,
    });
  }

  function loadExperiment(item: { prompt: string; temperature: number; topK: number }) {
    setPrompt(item.prompt);
    setDraft(item.prompt);
    setTemperature(item.temperature);
    setTopK(item.topK);
    setGenerated([]);
    setView("lab");
    toast.success("已恢复输入与解码参数");
    window.setTimeout(() => labRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }

  return (
    <div className="app-shell">
      <Toaster position="bottom-center" richColors closeButton />
      <header className="site-header">
        <button type="button" className="brand" onClick={() => switchView("lab")} aria-label="返回推理实验室">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>INFERENCE</strong><small>推理实验室</small></span>
        </button>
        <nav className="main-nav" aria-label="主导航">
          <button type="button" className={view === "lab" ? "is-active" : ""} onClick={() => switchView("lab")}><FlaskConical />实验台</button>
          <button type="button" className={view === "kv" ? "is-active" : ""} onClick={() => switchView("kv")}><Database />KV Cache</button>
          <button type="button" className={view === "attention" ? "is-active" : ""} onClick={() => switchView("attention")}><Network />注意力网络</button>
          <button type="button" className={view === "concepts" ? "is-active" : ""} onClick={() => switchView("concepts")}><BookOpen />概念速查</button>
          <button type="button" className={view === "notebook" ? "is-active" : ""} onClick={() => switchView("notebook")}><FolderClock />我的实验</button>
        </nav>
        <div className="account-area">
          {IS_STATIC_DEMO ? <span className="account-loading"><FlaskConical />静态演示</span> : authLoading ? <span className="account-loading"><LoaderCircle className="spin" />账户检查中</span> : user ? (
            <button type="button" className="account-button" onClick={logout} title="点击退出登录"><UserRound /><span>{user.name ?? "已登录"}</span><small>退出</small></button>
          ) : (
            <Link to="/login" className="account-button"><LogIn /><span>登录保存</span></Link>
          )}
        </div>
      </header>

      {view === "lab" && (
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
                  <input id="prompt" value={draft} maxLength={80} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") runPrompt(); }} aria-describedby={inputError ? "prompt-error" : undefined} />
                  <button type="button" className="button primary" onClick={() => runPrompt()}><Play />运行推理</button>
                </div>
                {inputError && <p className="field-error" id="prompt-error">{inputError}</p>}
                <div className="example-row"><span>试试</span>{EXAMPLES.map((example) => <button type="button" key={example} onClick={() => runPrompt(example)}>{example}</button>)}</div>
              </div>
              <div className="decode-controls">
                <label><span>Temperature <strong>{temperature.toFixed(1)}</strong></span><input type="range" min="0.1" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
                <label><span>Top-k <strong>{topK}</strong></span><input type="range" min="1" max="16" step="1" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></label>
                {!IS_STATIC_DEMO && <button type="button" className="button secondary save-button" onClick={isAuthenticated ? saveCurrent : () => switchView("notebook")} disabled={createExperiment.isPending}>{createExperiment.isPending ? <LoaderCircle className="spin" /> : <Save />}{isAuthenticated ? "保存实验" : "登录后保存"}</button>}
              </div>
            </div>

            <div className="generation-status" aria-live="polite">
              <span className="status-label">当前上下文</span>
              <div className="status-tokens">{trace.tokens.map((token, index) => <span className={index >= trace.tokens.length - generated.length && generated.length > 0 ? "is-generated" : ""} key={`${token.text}-${index}`}>{token.text}</span>)}</div>
              <small>{trace.tokens.length}/{MODEL_CONFIG.maxTokens} positions · 已生成 {generated.length} tokens</small>
            </div>

            <TransformerWorkbench
              key={prompt}
              trace={trace}
              candidates={candidates}
              selectedCandidate={selectedCandidate}
              onSelectCandidate={(candidate) => setSelectedCandidateId(candidate.id)}
              onCommitCandidate={commitCandidate}
              generatedCount={generated.length}
              canGenerate={canGenerate}
              temperature={temperature}
              topK={topK}
            />
          </section>
          <footer className="site-footer"><span>INFERENCE LAB</span><p>教学模型的输出不代表真实语言能力；它的价值是让每个算子、形状和状态变化都可见。</p><button type="button" onClick={() => switchView("concepts")}>继续理解 KV Cache<ArrowRight /></button></footer>
        </main>
      )}

      {view === "kv" && <KVCacheReplay />}

      {view === "attention" && <AttentionNetwork />}

      {view === "concepts" && <main className="subpage"><Concepts /></main>}

      {view === "notebook" && (
        <main className="subpage notebook-page">
          <header className="view-intro"><span className="stage-kicker">YOUR NOTEBOOK</span><h1>把一次观察，留成可重开的实验</h1><p>{IS_STATIC_DEMO ? "GitHub Pages 仅提供浏览器内的推理演示，不连接账户、数据库或服务端实验记录。" : "保存输入、生成结果与解码参数。记录保存在服务端，刷新或更换设备后仍可访问。"}</p></header>
          {IS_STATIC_DEMO ? (
            <div className="notebook-empty"><FolderClock /><h2>静态演示不保存实验记录</h2><p>所有模型计算仍可在本地浏览器中完整体验；登录与保存功能需要运行项目自带的 Hono 服务和 MySQL 数据库。</p><button type="button" className="button primary" onClick={() => switchView("lab")}>返回推理实验<ArrowRight /></button></div>
          ) : !isAuthenticated ? (
            <div className="notebook-empty"><FolderClock /><h2>{authLoading ? "正在检查登录状态" : "登录后查看你的实验记录"}</h2><p>推理实验室无需登录即可使用；只有个人保存功能需要账户。</p>{!authLoading && <Link className="button primary" to="/login"><LogIn />使用 Kimi 登录</Link>}</div>
          ) : (
            <section className="notebook-content">
              <div className="notebook-actions"><div><span className="micro-label">已登录</span><strong>{user?.name ?? "实验者"}</strong></div><button type="button" className="button primary" onClick={() => switchView("lab")}><Sparkles />新建实验</button></div>
              {experiments.isLoading ? <div className="notebook-empty"><LoaderCircle className="spin" /><h2>正在读取实验记录</h2></div> : experiments.isError ? <div className="notebook-empty"><CircleStop /><h2>记录暂时无法读取</h2><p>请检查网络后重试。</p><button type="button" className="button secondary" onClick={() => experiments.refetch()}>重新加载</button></div> : experiments.data?.length ? (
                <div className="experiment-list">{experiments.data.map((item) => (
                  <article className="experiment-item" key={item.id}>
                    <div className="experiment-meta"><span>{new Date(item.createdAt).toLocaleString("zh-CN")}</span><small>T {item.temperature.toFixed(1)} · Top-k {item.topK}</small></div>
                    <h2>{item.title}</h2><p><strong>输入</strong>{item.prompt}</p><p><strong>输出</strong>{item.output}</p>
                    <div className="experiment-actions"><button type="button" className="button secondary" onClick={() => loadExperiment(item)}>重新打开<ArrowRight /></button><button type="button" className="icon-button danger" aria-label={`删除 ${item.title}`} onClick={() => { if (window.confirm("确定删除这条实验记录吗？")) deleteExperiment.mutate({ id: item.id }); }}><Trash2 /></button></div>
                  </article>
                ))}</div>
              ) : <div className="notebook-empty"><FolderClock /><h2>还没有保存的实验</h2><p>回到实验台，运行一次推理后点击“保存实验”。</p><button type="button" className="button primary" onClick={() => switchView("lab")}>去做第一个实验<ArrowRight /></button></div>}
            </section>
          )}
        </main>
      )}
    </div>
  );
}
