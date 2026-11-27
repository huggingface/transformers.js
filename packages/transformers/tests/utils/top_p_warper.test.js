import { Tensor } from "../../src/utils/tensor.js";
import { TopPLogitsWarper } from "../../src/generation/logits_process.js";

import { init } from "../init.js";
init();

describe("TopPLogitsWarper", () => {
  it("keeps the smallest set of tokens whose probabilities sum to top_p", () => {
    // Softmax of [0, 1, 2, 3] is about [0.032, 0.087, 0.237, 0.644].
    const logits = new Tensor("float32", [0, 1, 2, 3], [1, 4]);
    const warper = new TopPLogitsWarper(0.7);
    warper([], logits);

    const data = Array.from(logits.data);
    // 0.644 + 0.237 = 0.881 >= 0.7, so only the two most likely tokens remain.
    expect(data[3]).toBe(3);
    expect(data[2]).toBe(2);
    expect(data[1]).toBe(-Infinity);
    expect(data[0]).toBe(-Infinity);
  });

  it("with a very small top_p, only the most likely token remains", () => {
    const logits = new Tensor("float32", [0, 5, 1], [1, 3]);
    const warper = new TopPLogitsWarper(1e-6);
    warper([], logits);
    expect(Array.from(logits.data)).toEqual([-Infinity, 5, -Infinity]);
  });
});
