import { Suspense, lazy } from 'react'
import { Navigate, Routes, Route } from 'react-router'
import Home from './pages/Home'
import NotFound from "./pages/NotFound"
import LabView from "@/views/LabView"

// 实验台视图随首屏直接打包（lab 永不懒加载）；其余视图按需加载，空闲时预取见 pages/Home.tsx。
const KVCacheReplay = lazy(() => import("@/views/kv/KVCacheReplay"));
const AttentionNetwork = lazy(() => import("@/views/attention/AttentionNetwork"));
const FlashAttentionExplorer = lazy(() => import("@/views/flash/FlashAttentionExplorer"));
const MoEExplorer = lazy(() => import("@/views/moe/MoEExplorer"));
const Concepts = lazy(() => import("@/views/concepts/Concepts"));

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />}>
        <Route index element={<LabView />} />
        <Route path="kv" element={<Suspense fallback={null}><KVCacheReplay /></Suspense>} />
        <Route path="attention" element={<Suspense fallback={null}><AttentionNetwork /></Suspense>} />
        <Route path="flash" element={<Suspense fallback={null}><FlashAttentionExplorer /></Suspense>} />
        <Route path="moe" element={<Suspense fallback={null}><MoEExplorer /></Suspense>} />
        <Route path="concepts" element={<main className="subpage"><Suspense fallback={null}><Concepts /></Suspense></main>} />
      </Route>
      <Route path="/login" element={<Navigate to="/" replace />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
