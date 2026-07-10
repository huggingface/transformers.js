import { jest } from "@jest/globals";
import { Tensor } from "@huggingface/transformers";

const computeMask = jest.fn();
const commitToken = jest.fn();
const createInterpreter = jest.fn(() => ({ computeMask, commitToken }));
const loadBundledLLGuidance = jest.fn(async () => ({ createInterpreter }));

jest.unstable_mockModule("llguidance", () => ({
  loadBundledLLGuidance,
}));

const { LlguidanceConstraint } = await import("../dist/index.js");

describe("LlguidanceConstraint", () => {
  beforeEach(() => {
    computeMask.mockReset();
    commitToken.mockReset();
    createInterpreter.mockClear();
    loadBundledLLGuidance.mockClear();
  });

  it("loads llguidance and applies masks", async () => {
    computeMask.mockReturnValue({ mask: [true, false, true, false], vocabSize: 4 });

    const tokenizer = { name: "tokenizer" };
    const response_format = { type: "json_schema", json_schema: { type: "object" } };
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat(tokenizer, response_format);
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 4]);

    logits_processor([[0n], [0n]], logits);

    expect(loadBundledLLGuidance).toHaveBeenCalledTimes(1);
    expect(createInterpreter).toHaveBeenCalledWith({ tokenizer, response_format });
    expect(Array.from(logits.data)).toEqual([1, -Infinity, 3, -Infinity, 5, -Infinity, 7, -Infinity]);
  });

  it("passes explicit llguidance load options", async () => {
    await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" }, { useWasmCache: false, wasmUrl: "custom.wasm" });

    expect(loadBundledLLGuidance).toHaveBeenCalledWith({ wasmUrl: "custom.wasm" });
  });

  it("commits sampled tokens and stops when llguidance stops", async () => {
    computeMask.mockReturnValue({ mask: [true, true], vocabSize: 2 });
    commitToken.mockReturnValueOnce(undefined).mockReturnValueOnce({ stop: true });

    const { logits_processor, stopping_criteria } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });

    expect(stopping_criteria([[0n]])).toEqual([false]);

    logits_processor.onTokensSampled([0, 1], [[0n, 0n], [0n, 1n]]);

    expect(commitToken).toHaveBeenCalledWith(0);
    expect(commitToken).toHaveBeenCalledWith(1);
    expect(stopping_criteria([[0n, 0n], [0n, 1n]])).toEqual([true, true]);
  });
});
