/**
 * 共享的文档目录（doc-outline）交互逻辑：滚动高亮、折叠/展开、点击跳转与（可选）hash 深链恢复。
 * 原先 MoE 与 FlashAttention 两页各持一份等价实现，此处合并，仅有的三处行为差异经 opts 开关保留：
 * - ariaLabel：折叠按钮随开合同步 aria-label（FlashAttention 页现状，MoE 页无此行为）
 * - ariaCurrent：激活链接同步 aria-current="location"（FlashAttention 页现状，MoE 页只切 active 类）
 * - hashRestore：挂载后按 location.hash 深链定位到对应小节（MoE 页现状，FlashAttention 页无）
 * 监听统一经 signal（AbortSignal）注销；原 MoE 版在清理函数里重复 removeEventListener，与 abort 语义等价，不再保留。
 */
export function setupDocOutline(root, opts = {}) {
  const { signal, ariaLabel = false, ariaCurrent = false, hashRestore = false } = opts;
  const byId = id => root.querySelector("#" + id);
  const outline = byId("docOutline");
  const toggle = byId("docToggle");
  const links = [...root.querySelectorAll('#docTree a[href^="#"]')];
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  function setOutline(open) {
    outline.classList.toggle("collapsed", !open);
    toggle.setAttribute("aria-expanded", String(open));
    if (ariaLabel) toggle.setAttribute("aria-label", open ? "折叠文档目录" : "展开文档目录");
    toggle.querySelector(".doc-chevron").textContent = open ? "−" : "＋";
    root.classList.toggle(
      "outline-docked",
      open && window.innerWidth >= 1720
    );
  }
  function setActive(id) {
    links.forEach(link => {
      const active = link.getAttribute("href") === "#" + id;
      link.classList.toggle("active", active);
      if (ariaCurrent) {
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      }
    });
  }
  function goTo(link) {
    const id = decodeURIComponent(link.getAttribute("href").slice(1));
    const target = byId(id);
    if (!target) return;
    setActive(id);
    if (window.innerWidth < 1720) setOutline(false);
    target.scrollIntoView({
      behavior: reduced.matches ? "auto" : "smooth",
      block: "start",
    });
    if (history.replaceState) history.replaceState(null, "", "#" + id);
  }
  links.forEach(link =>
    link.addEventListener(
      "click",
      event => {
        event.preventDefault();
        goTo(link);
      },
      { signal }
    )
  );
  toggle.addEventListener(
    "click",
    () => setOutline(outline.classList.contains("collapsed")),
    { signal }
  );
  let disposed = false;
  let ticking = false;
  function updateActive() {
    if (disposed) return;
    const y = window.scrollY + 180;
    let current = links[0].getAttribute("href").slice(1);
    for (const link of links) {
      const target = byId(link.getAttribute("href").slice(1));
      if (target && target.getBoundingClientRect().top + window.scrollY <= y)
        current = target.id;
      else break;
    }
    setActive(current);
    ticking = false;
  }
  const onScroll = () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(updateActive);
    }
  };
  const onResize = () => {
    root.classList.toggle(
      "outline-docked",
      !outline.classList.contains("collapsed") && window.innerWidth >= 1720
    );
  };
  window.addEventListener("scroll", onScroll, { passive: true, signal });
  window.addEventListener("resize", onResize, { signal });
  setOutline(window.innerWidth >= 1720);
  updateActive();
  if (hashRestore && location.hash) {
    const target = byId(decodeURIComponent(location.hash.slice(1)));
    if (target)
      requestAnimationFrame(() => {
        setActive(target.id);
        target.scrollIntoView({ block: "start" });
      });
  }
  return () => {
    disposed = true;
  };
}
