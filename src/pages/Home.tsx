import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Outlet, useLocation, useNavigate } from "react-router";
import {
  BookOpen, Database, FlaskConical, Network, Waypoints, Zap,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import "../App.css";
import {
  MODEL_CONFIG, forward, getCandidates, tokenize,
  type Candidate, type ForwardTrace, type Token,
} from "@/lib/tinyTransformer";

const EXAMPLES = ["小猫坐在垫子上", "机器人站在沙发旁边", "小狗躺在草地上"];

/** 教学页面对应 HomeShell 下的子路由（可深链、可回退），视图名用于导航与激活态判定。 */
export type HomeView = "lab" | "kv" | "attention" | "flash" | "moe" | "concepts";

const VIEW_PATHS: Record<HomeView, string> = {
  lab: "/",
  kv: "/kv",
  attention: "/attention",
  flash: "/flash",
  moe: "/moe",
  concepts: "/concepts",
};

const PATH_VIEWS: Record<string, Exclude<HomeView, "lab">> = {
  "/kv": "kv",
  "/attention": "attention",
  "/flash": "flash",
  "/moe": "moe",
  "/concepts": "concepts",
};

function viewFromPath(pathname: string): HomeView {
  const normalized = pathname.length > 1 && pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
  return PATH_VIEWS[normalized] ?? "lab";
}

/** 经 <Outlet context> 下发给子路由视图（LabView）的状态与能力。 */
export type HomeOutletContext = {
  prompt: string;
  trace: ForwardTrace;
  candidates: Candidate[];
  selectedCandidate: Candidate;
  generatedCount: number;
  canGenerate: boolean;
  temperature: number;
  topK: number;
  labRef: RefObject<HTMLElement | null>;
  examples: string[];
  draft: string;
  onDraftChange: (value: string) => void;
  inputError: string;
  onRunPrompt: (prompt?: string) => void;
  onTemperatureChange: (value: number) => void;
  onTopKChange: (value: number) => void;
  onSelectCandidate: (candidate: Candidate) => void;
  onCommitCandidate: (candidate: Candidate) => void;
  onSwitchView: (view: HomeView) => void;
};

export default function HomeShell() {
  const [draft, setDraft] = useState(EXAMPLES[0]);
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [generated, setGenerated] = useState<Token[]>([]);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(8);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const [inputError, setInputError] = useState("");
  const labRef = useRef<HTMLElement>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const view = viewFromPath(location.pathname);

  const trace = useMemo(() => forward([...tokenize(prompt), ...generated]), [prompt, generated]);
  const candidates = useMemo(
    () => getCandidates(trace.logits, temperature, topK),
    [trace.logits, temperature, topK],
  );
  const selectedCandidate = candidates.find((item) => item.id === selectedCandidateId) ?? candidates[0];
  const canGenerate = trace.tokens.length < MODEL_CONFIG.maxTokens && trace.tokens.at(-1)?.id !== 1;

  useEffect(() => {
    // 子路由的懒加载分块在 App.tsx 声明，这里只负责空闲时预取（lab 随首屏打包，无需预取）。
    const prefetch = () => {
      void import("@/views/kv/KVCacheReplay");
      void import("@/views/attention/AttentionNetwork");
      void import("@/views/flash/FlashAttentionExplorer");
      void import("@/views/moe/MoEExplorer");
      void import("@/views/concepts/Concepts");
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

  function switchView(next: HomeView) {
    // 声明式导航不会自动滚动，平滑滚回顶部保留在点击处理里；重复点击当前视图不产生历史记录。
    if (view !== next) navigate(VIEW_PATHS[next]);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const outletContext: HomeOutletContext = {
    prompt,
    trace,
    candidates,
    selectedCandidate,
    generatedCount: generated.length,
    canGenerate,
    temperature,
    topK,
    labRef,
    examples: EXAMPLES,
    draft,
    onDraftChange: setDraft,
    inputError,
    onRunPrompt: runPrompt,
    onTemperatureChange: setTemperature,
    onTopKChange: setTopK,
    onSelectCandidate: (candidate) => setSelectedCandidateId(candidate.id),
    onCommitCandidate: commitCandidate,
    onSwitchView: switchView,
  };

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
        </nav>
      </header>

      <Outlet context={outletContext} />
    </div>
  );
}
