import { useEffect, useRef } from "react";
import markup from "./moe-explorer.html?raw";
import { initializeMoEExplorer } from "./moe-explorer.js";
import "./MoEExplorer.css";

export default function MoEExplorer() {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!rootRef.current) return;
    return initializeMoEExplorer(rootRef.current);
  }, []);

  return (
    <div
      ref={rootRef}
      className="moe-explorer-page"
      aria-label="MoE 计算、路由与成本交互图解"
      dangerouslySetInnerHTML={{ __html: markup }}
    />
  );
}
