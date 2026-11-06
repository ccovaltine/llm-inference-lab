import { Link, useOutletContext } from "react-router";
import { ArrowRight, CircleStop, FolderClock, LoaderCircle, LogIn, Sparkles, Trash2 } from "lucide-react";
import { IS_STATIC_DEMO } from "@/const";
import type { HomeOutletContext } from "@/pages/Home";

export default function NotebookView() {
  const {
    user,
    authLoading,
    isAuthenticated,
    experiments,
    onLoadExperiment,
    onDeleteExperiment,
    onSwitchView,
  } = useOutletContext<HomeOutletContext>();
  return (
    <main className="subpage notebook-page">
      <header className="view-intro"><span className="stage-kicker">YOUR NOTEBOOK</span><h1>把一次观察，留成可重开的实验</h1><p>{IS_STATIC_DEMO ? "GitHub Pages 仅提供浏览器内的推理演示，不连接账户、数据库或服务端实验记录。" : "保存输入、生成结果与解码参数。记录保存在服务端，刷新或更换设备后仍可访问。"}</p></header>
      {IS_STATIC_DEMO ? (
        <div className="notebook-empty"><FolderClock /><h2>静态演示不保存实验记录</h2><p>所有模型计算仍可在本地浏览器中完整体验；登录与保存功能需要运行项目自带的 Hono 服务和 MySQL 数据库。</p><button type="button" className="button primary" onClick={() => onSwitchView("lab")}>返回推理实验<ArrowRight /></button></div>
      ) : !isAuthenticated ? (
        <div className="notebook-empty"><FolderClock /><h2>{authLoading ? "正在检查登录状态" : "登录后查看你的实验记录"}</h2><p>推理实验室无需登录即可使用；只有个人保存功能需要账户。</p>{!authLoading && <Link className="button primary" to="/login"><LogIn />使用 Kimi 登录</Link>}</div>
      ) : (
        <section className="notebook-content">
          <div className="notebook-actions"><div><span className="micro-label">已登录</span><strong>{user?.name ?? "实验者"}</strong></div><button type="button" className="button primary" onClick={() => onSwitchView("lab")}><Sparkles />新建实验</button></div>
          {experiments.isLoading ? <div className="notebook-empty"><LoaderCircle className="spin" /><h2>正在读取实验记录</h2></div> : experiments.isError ? <div className="notebook-empty"><CircleStop /><h2>记录暂时无法读取</h2><p>请检查网络后重试。</p><button type="button" className="button secondary" onClick={() => experiments.refetch()}>重新加载</button></div> : experiments.data?.length ? (
            <div className="experiment-list">{experiments.data.map((item) => (
              <article className="experiment-item" key={item.id}>
                <div className="experiment-meta"><span>{new Date(item.createdAt).toLocaleString("zh-CN")}</span><small>T {item.temperature.toFixed(1)} · Top-k {item.topK}</small></div>
                <h2>{item.title}</h2><p><strong>输入</strong>{item.prompt}</p><p><strong>输出</strong>{item.output}</p>
                <div className="experiment-actions"><button type="button" className="button secondary" onClick={() => onLoadExperiment(item)}>重新打开<ArrowRight /></button><button type="button" className="icon-button danger" aria-label={`删除 ${item.title}`} onClick={() => { if (window.confirm("确定删除这条实验记录吗？")) onDeleteExperiment(item.id); }}><Trash2 /></button></div>
              </article>
            ))}</div>
          ) : <div className="notebook-empty"><FolderClock /><h2>还没有保存的实验</h2><p>回到实验台，运行一次推理后点击“保存实验”。</p><button type="button" className="button primary" onClick={() => onSwitchView("lab")}>去做第一个实验<ArrowRight /></button></div>}
        </section>
      )}
    </main>
  );
}
