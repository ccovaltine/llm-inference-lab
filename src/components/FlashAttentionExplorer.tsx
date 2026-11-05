import { useEffect, useRef } from "react";
import markup from "./flash-attention-explorer.html?raw";
import { initializeFlashAttentionExplorer } from "./flash-attention-explorer.js";
import "./FlashAttentionExplorer.css";

export default function FlashAttentionExplorer() {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    root.innerHTML = markup;
    return initializeFlashAttentionExplorer(root);
  }, []);

  return (
    <div
      ref={rootRef}
      className="flash-attention-page"
      aria-label="FlashAttention 分块计算、访存与 Online Softmax 交互图解"
    />
  );
}
