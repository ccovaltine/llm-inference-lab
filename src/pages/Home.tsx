import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Link, Outlet, useLocation, useNavigate } from "react-router";
import {
  BookOpen, Database, FlaskConical, FolderClock, LoaderCircle,
  LogIn, Network, UserRound, Waypoints, Zap,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import type { inferRouterOutputs } from "@trpc/server";
import type { AppRouter } from "../../api/router";
import "../App.css";
import { trpc } from "@/providers/trpc";
import { useAuth } from "@/hooks/useAuth";
import { IS_STATIC_DEMO } from "@/const";
import {
  MODEL_CONFIG, forward, getCandidates, tokenize,
  type Candidate, type ForwardTrace, type Token,
} from "@/lib/tinyTransformer";

type ExperimentItem = inferRouterOutputs<AppRouter>["experiment"]["list"][number];

const EXAMPLES = ["小猫坐在垫子上", "机器人站在沙发旁边", "小狗躺在草地上"];

/** 教学页面对应 HomeShell 下的子路由（可深链、可回退），视图名用于导航与激活态判定。 */
export type HomeView = "lab" | "kv" | "attention" | "flash" | "moe" | "concepts" | "notebook";

const VIEW_PATHS: Record<HomeView, string> = {
  lab: "/",
  kv: "/kv",
  attention: "/attention",
  flash: "/flash",
  moe: "/moe",
  concepts: "/concepts",
  notebook: "/notebook",
};

const PATH_VIEWS: Record<string, Exclude<HomeView, "lab">> = {
  "/kv": "kv",
  "/attention": "attention",
  "/flash": "flash",
  "/moe": "moe",
  "/concepts": "concepts",
  "/notebook": "notebook",
};

function viewFromPath(pathname: string): HomeView {
  const normalized = pathname.length > 1 && pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
  return PATH_VIEWS[normalized] ?? "lab";
}

/** 经 <Outlet context> 下发给子路由视图（LabView / NotebookView）的状态与能力。 */
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
  isAuthenticated: boolean;
  createPending: boolean;
  onSaveCurrent: () => void;
  onSwitchView: (view: HomeView) => void;
  user: { name?: string | null } | null | undefined;
  authLoading: boolean;
  experiments: {
    isLoading: boolean;
    isError: boolean;
    refetch: () => void;
    data?: ExperimentItem[];
  };
  onLoadExperiment: (item: { prompt: string; temperature: number; topK: number }) => void;
  onDeleteExperiment: (id: number) => void;
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
    // 子路由的懒加载分块在 App.tsx 声明，这里只负责空闲时预取（lab 随首屏打包，无需预取）。
    const prefetch = () => {
      void import("@/views/kv/KVCacheReplay");
      void import("@/views/attention/AttentionNetwork");
      void import("@/views/flash/FlashAttentionExplorer");
      void import("@/views/moe/MoEExplorer");
      void import("@/views/concepts/Concepts");
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

  function switchView(next: HomeView) {
    // 声明式导航不会自动滚动，平滑滚回顶部保留在点击处理里；重复点击当前视图不产生历史记录。
    if (view !== next) navigate(VIEW_PATHS[next]);
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
    navigate("/");
    toast.success("已恢复输入与解码参数");
    window.setTimeout(() => labRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
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
    isAuthenticated,
    createPending: createExperiment.isPending,
    onSaveCurrent: saveCurrent,
    onSwitchView: switchView,
    user,
    authLoading,
    experiments,
    onLoadExperiment: loadExperiment,
    onDeleteExperiment: (id) => deleteExperiment.mutate({ id }),
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

      <Outlet context={outletContext} />
    </div>
  );
}
