/**
 * 跨页面共享的纯数值工具。
 * 语义与各页面原先的本地实现逐位一致：合并前已逐行比对。
 */

/**
 * 数值稳定的逐行 softmax：
 * 减去最大有限值防溢出；非有限输入（如被 Top-k 置为 −∞ 的位置）按 0 处理。
 * 原先两份等价实现：tinyTransformer.ts 的 softmax / inferenceOperations.ts 的 softmaxRow。
 */
export function softmax(values: number[]): number[] {
  const max = Math.max(...values.filter(Number.isFinite));
  const exps = values.map((value) => (Number.isFinite(value) ? Math.exp(value - max) : 0));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}
