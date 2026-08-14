import { jest } from "@jest/globals";
import { Tensor } from "@huggingface/transformers";

const computeMask = jest.fn();
const computeMaskInto = jest.fn();
const validateCandidates = jest.fn();
const computeFastForward = jest.fn();
const commitToken = jest.fn();
const disposeInterpreter = jest.fn();
const createInterpreter = jest.fn(() => ({
  computeMask,
  computeMaskInto,
  validateCandidates,
  computeFastForward,
  commitToken,
  dispose: disposeInterpreter,
}));
const loadBundledLLGuidance = jest.fn(async () => ({ createInterpreter }));

jest.unstable_mockModule("llguidance", () => ({
  loadBundledLLGuidance,
}));

const { LlguidanceConstraint } = await import("../dist/index.js");

describe("LlguidanceConstraint", () => {
  beforeEach(() => {
    computeMask.mockReset();
    computeMaskInto.mockReset();
    validateCandidates.mockReset();
    computeFastForward.mockReset();
    commitToken.mockReset();
    disposeInterpreter.mockReset();
    createInterpreter.mockClear();
    loadBundledLLGuidance.mockClear();

    computeMaskInto.mockImplementation((target) => {
      target[0] = 0b0101;
      return { mask: target, vocabSize: 4 };
    });
    commitToken.mockReturnValue({ stop: false, backtrack: 0, ffTokens: [] });
  });

  it("loads llguidance and applies a packed mask", async () => {
    const tokenizer = { name: "tokenizer" };
    const response_format = { type: "json_schema", json_schema: { type: "object" } };
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat(tokenizer, response_format);
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);

    logits_processor([[0n]], logits);

    expect(loadBundledLLGuidance).toHaveBeenCalledTimes(1);
    expect(createInterpreter).toHaveBeenCalledWith({ tokenizer, response_format });
    expect(Array.from(logits.data)).toEqual([1, -Infinity, 3, -Infinity]);
  });

  it("reuses the packed mask buffer", async () => {
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const first = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);
    const second = new Tensor("float32", new Float32Array([5, 6, 7, 8]), [1, 4]);

    logits_processor([[0n]], first);
    logits_processor([[0n]], second);

    expect(computeMask).not.toHaveBeenCalled();
    expect(computeMaskInto).toHaveBeenCalledTimes(2);
    expect(computeMaskInto.mock.calls[0][0]).toBe(computeMaskInto.mock.calls[1][0]);
  });

  it("exposes the packed mask to native generation plans", async () => {
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const native = logits_processor.processors[0].getRuntimeGenerationProcessor();

    expect(native.op).toBe("token-mask");
    expect(Array.from(native.getMask(4))).toEqual([0b0101]);
    expect(computeMaskInto).toHaveBeenCalledTimes(1);
  });

  it("exposes candidate validation and fast-forward to native plans", async () => {
    const validity = Uint8Array.of(1, 0);
    validateCandidates.mockReturnValue(validity);
    computeFastForward.mockReturnValue([3, 4]);
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const native = logits_processor.processors[0].getRuntimeGenerationProcessor();
    const ids = Uint32Array.of(3, 9);

    expect(native.validateCandidates(ids)).toBe(validity);
    expect(validateCandidates).toHaveBeenCalledWith(ids, undefined);
    expect(native.getFastForward(2)).toEqual([3, 4]);
    expect(computeFastForward).toHaveBeenCalledWith(2);
    expect(commitToken).not.toHaveBeenCalled();
  });

  it("rejects batched generation before computing a mask", async () => {
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array(8), [2, 4]);

    expect(() => logits_processor([[0n], [0n]], logits)).toThrow("currently supports batch size 1");
    expect(computeMaskInto).not.toHaveBeenCalled();
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("commits one sampled token and stops on acceptance", async () => {
    commitToken.mockReturnValue({ stop: true, backtrack: 0, ffTokens: [] });
    const { logits_processor, stopping_criteria, stats } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });

    expect(stopping_criteria([[0n]])).toEqual([false]);
    logits_processor.onTokensSampled([1], [[0n, 1n]]);

    expect(commitToken).toHaveBeenCalledWith(1);
    expect(stopping_criteria([[0n, 1n]])).toEqual([true]);
    expect(stats.stopReason).toBe("accepted");
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("throws when llguidance requests backtracking", async () => {
    commitToken.mockReturnValue({ stop: false, backtrack: 1, ffTokens: [] });
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });

    expect(() => logits_processor.onTokensSampled([1], [[0n, 1n]])).toThrow("requested backtracking by 1 token");
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("records and throws on a dead end", async () => {
    computeMaskInto.mockReturnValue({ stop: true, reason: "dead_end" });
    const { logits_processor, stats } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array(4), [1, 4]);

    expect(() => logits_processor([[0n]], logits)).toThrow("reached a dead end");
    expect(stats.stopReason).toBe("dead_end");
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("stops and disposes when computeMask reports acceptance", async () => {
    computeMaskInto.mockReturnValue({ stop: true, reason: "accepted" });
    const { logits_processor, stopping_criteria, stats } = await LlguidanceConstraint.fromResponseFormat({ eos_token_id: 1 }, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);

    logits_processor([[0n]], logits);
    logits_processor.onTokensSampled([0], [[0n]]);

    expect(Array.from(logits.data)).toEqual([-Infinity, 2, -Infinity, -Infinity]);
    expect(commitToken).not.toHaveBeenCalled();
    expect(stopping_criteria([[0n]])).toEqual([true]);
    expect(stats.stopReason).toBe("accepted");
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("fails closed on mask acceptance when the tokenizer has no EOS token ID", async () => {
    computeMaskInto.mockReturnValue({ stop: true, reason: "accepted" });
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array([1, 2, 3, 4]), [1, 4]);

    expect(() => logits_processor([[0n]], logits)).toThrow("tokenizer has no EOS token ID");
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("fails closed when logits have no vocabulary dimension", async () => {
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = { dims: [], data: new Float32Array(4) };

    expect(() => logits_processor([[0n]], logits)).toThrow("requires logits with a vocabulary dimension");
    expect(computeMaskInto).not.toHaveBeenCalled();
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("fails closed when mask vocabulary metadata is invalid", async () => {
    computeMaskInto.mockReturnValue({ mask: new Uint32Array([0b0101]), vocabSize: undefined });
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array(4), [1, 4]);

    expect(() => logits_processor([[0n]], logits)).toThrow("invalid vocabulary size");
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });

  it("exposes idempotent disposal", async () => {
    const { dispose, logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });

    dispose();
    dispose();

    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
    expect(() => logits_processor.onTokensSampled([1], [[0n, 1n]])).toThrow("has been disposed");
  });

  it("disposes and rethrows interpreter errors", async () => {
    const error = new Error("unexpected");
    computeMaskInto.mockImplementation(() => {
      throw error;
    });
    const { logits_processor } = await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });
    const logits = new Tensor("float32", new Float32Array(4), [1, 4]);

    expect(() => logits_processor([[0n]], logits)).toThrow(error);
    expect(disposeInterpreter).toHaveBeenCalledTimes(1);
  });
});
