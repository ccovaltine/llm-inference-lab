import { describe, expect, it } from "vitest";
import { forward, getCandidates } from "./tinyTransformer";
import { OPERATION_COUNT, buildExpansion, buildOperation } from "./inferenceOperations";

describe("detailed inference operation ledger", () => {
  const trace = forward("小猫坐在垫子上");
  const candidates = getCandidates(trace.logits, 0.8, 8);
  const selected = candidates[0];
  const operations = Array.from({ length: OPERATION_COUNT }, (_, index) =>
    buildOperation(index, trace, 0, 0.8, 8, selected),
  );

  it("exposes real inputs, parameters, outputs and a numeric expansion at every node", () => {
    expect(operations).toHaveLength(43);
    for (const operation of operations) {
      expect(operation.outputs.length).toBeGreaterThan(0);
      const expansion = buildExpansion(operation, 0, 0, trace);
      expect(expansion.expression.length).toBeGreaterThan(0);
      expect(expansion.result).not.toContain("NaN");
      for (const item of [...operation.inputs, ...operation.parameters, ...operation.outputs]) {
        expect(item.data.length).toBeGreaterThan(0);
        expect(item.data[0].length).toBeGreaterThan(0);
      }
    }
  });

  it("uses the actual projection weight column and includes MLP bias", () => {
    const query = operations[5];
    const r = 2;
    const c = 3;
    const expected = query.inputs[0].data[r].reduce((sum, value, k) =>
      sum + value * query.parameters[0].data[k][c], 0);
    expect(query.outputs[0].data[r][c]).toBeCloseTo(expected, 10);
    expect(buildExpansion(query, r, c, trace).terms).toHaveLength(8);

    const linear = operations[16];
    const withBias = linear.inputs[0].data[r].reduce((sum, value, k) =>
      sum + value * linear.parameters[0].data[k][c], linear.parameters[1].data[0][c]);
    expect(linear.outputs[0].data[r][c]).toBeCloseTo(withBias, 10);
    expect(buildExpansion(linear, r, c, trace).terms).toHaveLength(9);
  });

  it("separates raw scores, causal mask, softmax and context", () => {
    const raw = operations[8].outputs[0].data;
    const masked = operations[9].outputs[0].data;
    const attention = operations[10].outputs[0].data;
    expect(Number.isFinite(raw[0][1])).toBe(true);
    expect(masked[0][1]).toBe(Number.NEGATIVE_INFINITY);
    expect(attention[0][1]).toBe(0);
    expect(attention[3].reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    expect(operations[11].outputs[0].data[0]).toHaveLength(4);
  });

  it("keeps full-vocabulary decoding states through top-k and softmax", () => {
    const clipped = operations[39].outputs[0].data[0];
    const probabilities = operations[40].outputs[0].data[0];
    expect(clipped.filter(Number.isFinite)).toHaveLength(8);
    expect(probabilities).toHaveLength(29);
    expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    expect(operations[37].parameters[0].data).toHaveLength(29);
    expect(operations[37].parameters[0].data[0]).toHaveLength(8);
  });
});
