import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import {
  BookOpen, Database, FlaskConical, FolderClock, LoaderCircle,
  LogIn, Network, UserRound, Waypoints, Zap,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import "../App.css";
import { trpc } from "@/providers/trpc";
import { useAuth } from "@/hooks/useAuth";
import { IS_STATIC_DEMO } from "@/const";
import LabView from "@/views/LabView";
import {
  MODEL_CONFIG, forward, getCandidates, tokenize,
  type Candidate, type Token,
} from "@/lib/tinyTransformer";

// 实验台视图随首屏直接打包（lab 永不懒加载）；其余视图按需加载并在空闲时预取。
const KVCacheReplay = lazy(() => import("@/components/KVCacheReplay"));
const AttentionNetwork = lazy(() => import("@/components/AttentionNetwork"));
const FlashAttentionExplorer = lazy(() => import("@/components/FlashAttentionExplorer"));
const MoEExplorer = lazy(() => import("@/components/MoEExplorer"));
const Concepts = lazy(() => import("@/components/Concepts"));
const NotebookView = lazy(() => import("@/views/NotebookView"));

const EXAMPLES = ["小猫坐在垫子上", "机器人站在沙发旁边", "小狗躺在草地上"];
type View = "lab" | "kv" | "attention" | "flash" | "moe" | "concepts" | "notebook";

export default function HomeShell() {
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

  useEffect(() => {
    const prefetch = () => {
      void import("@/components/KVCacheReplay");
      void import("@/components/AttentionNetwork");
      void import("@/components/FlashAttentionExplorer");
      void import("@/components/MoEExplorer");
      void import("@/components/Concepts");
      void import("@/views/NotebookView");
    };
    if (typeof window.requestIdleCallback === "function") {
      const handle = window.requestIdleCallback(prefetch);
      return () => window.cancelIdleCallback(handle);
    }
    const handle = window.setTimeout(prefetch, 200);
    return () => window.clearTimeout(handle);
  }, []);

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
          <button type="button" className={view === "flash" ? "is-active" : ""} onClick={() => switchView("flash")}><Zap />FlashAttention</button>
          <button type="button" className={view === "moe" ? "is-active" : ""} onClick={() => switchView("moe")}><Waypoints />MoE 网络</button>
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
        <LabView
          prompt={prompt}
          trace={trace}
          candidates={candidates}
          selectedCandidate={selectedCandidate}
          generatedCount={generated.length}
          canGenerate={canGenerate}
          temperature={temperature}
          topK={topK}
          labRef={labRef}
          examples={EXAMPLES}
          draft={draft}
          onDraftChange={setDraft}
          inputError={inputError}
          onRunPrompt={runPrompt}
          onTemperatureChange={setTemperature}
          onTopKChange={setTopK}
          onSelectCandidate={(candidate) => setSelectedCandidateId(candidate.id)}
          onCommitCandidate={commitCandidate}
          isAuthenticated={isAuthenticated}
          createPending={createExperiment.isPending}
          onSaveCurrent={saveCurrent}
          onSwitchView={switchView}
        />
      )}

      {view === "kv" && <Suspense fallback={null}><KVCacheReplay /></Suspense>}

      {view === "attention" && <Suspense fallback={null}><AttentionNetwork /></Suspense>}

      {view === "flash" && <Suspense fallback={null}><FlashAttentionExplorer /></Suspense>}

      {view === "moe" && <Suspense fallback={null}><MoEExplorer /></Suspense>}

      {view === "concepts" && <main className="subpage"><Suspense fallback={null}><Concepts /></Suspense></main>}

      {view === "notebook" && (
        <Suspense fallback={null}><NotebookView
          user={user}
          authLoading={authLoading}
          isAuthenticated={isAuthenticated}
          experiments={experiments}
          onLoadExperiment={loadExperiment}
          onDeleteExperiment={(id) => deleteExperiment.mutate({ id })}
          onSwitchView={switchView}
        /></Suspense>
      )}
    </div>
  );
}
