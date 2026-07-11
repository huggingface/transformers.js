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

  // Verifies the core integration path: llguidance creates an interpreter and its mask is applied across batches.
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

  // Ensures caller-provided load options are forwarded without the cache-only control flag.
  it("passes explicit llguidance load options", async () => {
    await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" }, { useWasmCache: false, wasmUrl: "custom.wasm" });

    expect(loadBundledLLGuidance).toHaveBeenCalledWith({ wasmUrl: "custom.wasm" });
  });

  // Confirms sampled tokens are committed back to llguidance so stopping criteria can end generation.
  it("commits sampled tokens and stops when llguidance stops", async () => {
    computeMask.mockReturnValue({ mask: [true, true], vocabSize: 2 });
    commitToken.mockReturnValueOnce(undefined).mockReturnValueOnce({ stop: true });

    const { logits_processor, stopping_criteria } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });

    expect(stopping_criteria([[0n]])).toEqual([false]);

    logits_processor.onTokensSampled(
      [0, 1],
      [
        [0n, 0n],
        [0n, 1n],
      ],
    );

    expect(commitToken).toHaveBeenCalledWith(0);
    expect(commitToken).toHaveBeenCalledWith(1);
    expect(
      stopping_criteria([
        [0n, 0n],
        [0n, 1n],
      ]),
    ).toEqual([true, true]);
  });

  // Covers llguidance's compact bitset mask format, not just boolean arrays.
  it("applies packed uint32 masks", async () => {
    computeMask.mockReturnValue({ mask: new Uint32Array([0b0101]), vocabSize: 4 });

    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);

    logits_processor([[0n]], logits);

    expect(Array.from(logits.data)).toEqual([1, -Infinity, 3, -Infinity]);
  });

  // Ensures a stop response from computeMask marks the shared state complete without committing more tokens.
  it("stops generation when computeMask returns stop", async () => {
    computeMask.mockReturnValue({ stop: true });

    const { logits_processor, stopping_criteria } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array([1, 2]), [1, 2]);

    logits_processor([[0n]], logits);
    logits_processor.onTokensSampled([0], [[0n]]);

    expect(Array.from(logits.data)).toEqual([1, 2]);
    expect(commitToken).not.toHaveBeenCalled();
    expect(stopping_criteria([[0n]])).toEqual([true]);
  });

  // Handles llguidance's post-stop compute error as a normal completion signal.
  it("treats compute_mask after stop as completed", async () => {
    computeMask.mockImplementation(() => {
      throw new Error("compute_mask() called after stop");
    });

    const { logits_processor, stopping_criteria } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array([1, 2]), [1, 2]);

    logits_processor([[0n]], logits);

    expect(Array.from(logits.data)).toEqual([1, 2]);
    expect(stopping_criteria([[0n]])).toEqual([true]);
  });

  // Keeps unexpected interpreter failures visible instead of swallowing real bugs.
  it("rethrows unexpected computeMask errors", async () => {
    const error = new Error("unexpected");
    computeMask.mockImplementation(() => {
      throw error;
    });

    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array([1, 2]), [1, 2]);

    expect(() => logits_processor([[0n]], logits)).toThrow(error);
  });
});
