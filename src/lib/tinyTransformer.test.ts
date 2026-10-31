import { describe, expect, it } from "vitest";
import { forward, getCandidates, MODEL_CONFIG, tokenize } from "./tinyTransformer";

describe("tiny transformer inference", () => {
  it("creates causal attention distributions with no future leakage", () => {
    const trace = forward("小猫坐在垫子上");
    for (const layer of trace.layers) {
      for (const head of layer.attention) {
        head.forEach((row, i) => {
          expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 8);
          row.forEach((value, j) => { if (j > i) expect(value).toBe(0); });
        });
      }
    }
  });

  it("keeps tensor shapes through both blocks and the LM head", () => {
    const trace = forward("机器人站在沙发旁边");
    expect(trace.embeddings[0]).toHaveLength(MODEL_CONFIG.dModel);
    expect(trace.layers).toHaveLength(2);
    expect(trace.layers[0].mlpHidden[0]).toHaveLength(MODEL_CONFIG.dFf);
    for (const layer of trace.layers) {
      expect(layer.scores[0][0][1]).toBe(Number.NEGATIVE_INFINITY);
      expect(layer.contexts).toHaveLength(2);
      expect(layer.concatenated[0]).toHaveLength(MODEL_CONFIG.dModel);
      expect(layer.mlpPreActivation[0]).toHaveLength(MODEL_CONFIG.dFf);
      expect(layer.mlpOutput[0]).toHaveLength(MODEL_CONFIG.dModel);
      for (let i = 0; i < trace.tokens.length; i += 1) {
        for (let d = 0; d < MODEL_CONFIG.dModel; d += 1) {
          expect(layer.afterAttention[i][d]).toBeCloseTo(layer.input[i][d] + layer.projected[i][d], 10);
          expect(layer.output[i][d]).toBeCloseTo(layer.afterAttention[i][d] + layer.mlpOutput[i][d], 10);
        }
      }
    }
    expect(trace.logits).toHaveLength(29);
    expect(trace.finalNormalized.at(-1)).toHaveLength(8);
  });

  it("re-normalizes after top-k filtering and responds to temperature", () => {
    const trace = forward("小狗躺在草地上");
    const sharp = getCandidates(trace.logits, 0.1, 8);
    const flat = getCandidates(trace.logits, 2, 8);
    expect(sharp.reduce((sum, item) => sum + item.probability, 0)).toBeCloseTo(1, 8);
    expect(sharp[0].probability).toBeGreaterThan(flat[0].probability);
  });

  it("marks out-of-vocabulary input instead of dropping it", () => {
    const tokens = tokenize("小猫hello");
    expect(tokens.at(-1)?.unknown).toBe(true);
    expect(tokens.at(-1)?.id).toBe(2);
  });
});
