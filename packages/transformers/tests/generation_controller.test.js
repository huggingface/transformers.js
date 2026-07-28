import { jest } from "@jest/globals";

import { AutoModel } from "../src/models/auto/modeling_auto.js";
import { GenerationController, createGenerationController } from "../src/generation/controller.js";
import { LogitsProcessor, LogitsProcessorList } from "../src/generation/logits_process.js";
import { StoppingCriteria, StoppingCriteriaList } from "../src/generation/stopping_criteria.js";
import { Tensor } from "../src/utils/tensor.js";

class ForceTokenProcessor extends LogitsProcessor {
  constructor(tokenId) {
    super();
    this.tokenId = tokenId;
  }

  _call(_inputIds, logits) {
    logits.data.fill(-Infinity);
    for (let batchIndex = 0; batchIndex < logits.dims[0]; ++batchIndex) {
      logits.data[batchIndex * logits.dims.at(-1) + this.tokenId] = 0;
    }
    return logits;
  }
}

class TokenStoppingCriteria extends StoppingCriteria {
  constructor(tokenId) {
    super();
    this.tokenId = BigInt(tokenId);
  }

  _call(inputIds) {
    return inputIds.map((tokens) => tokens.at(-1) === this.tokenId);
  }
}

function int64Tensor(values) {
  return new Tensor("int64", BigInt64Array.from(values.flat().map(BigInt)), [values.length, values[0].length]);
}

function createLease(values, release = jest.fn()) {
  return {
    version: 1,
    dtype: "float32",
    shape: [1, values.length],
    read: jest.fn(async () => Float32Array.from(values)),
    release,
  };
}

describe("GenerationController", () => {
  it("owns processing, sampling, stopping, streaming, and finalization", async () => {
    const processors = new LogitsProcessorList();
    processors.push(new ForceTokenProcessor(2));
    const criteria = new StoppingCriteriaList();
    criteria.push(new TokenStoppingCriteria(2));
    const streamer = { put: jest.fn(), end: jest.fn() };
    const model = {
      config: { eos_token_id: null },
      generation_config: null,
    };
    const controller = createGenerationController(model, int64Tensor([[1]]), { max_new_tokens: 4, logits_processor: processors, stopping_criteria: criteria, streamer });

    const step = await controller.step(new Tensor("float32", [10, 9, 0, 8], [1, 4]));

    expect(step.nextTokenIds.tolist()).toEqual([[2n]]);
    expect(step.allDone).toBe(true);
    expect(streamer.put.mock.calls).toEqual([[[[1n]]], [[[2n]]]]);
    expect(controller.finalize().tolist()).toEqual([[1n, 2n]]);
    expect(streamer.end).toHaveBeenCalledTimes(1);
  });

  it("supports zero-token generation without a model step", () => {
    const streamer = { put: jest.fn(), end: jest.fn() };
    const controller = createGenerationController({ config: {}, generation_config: null }, int64Tensor([[1, 2]]), { max_new_tokens: 0, streamer });

    expect(controller.allDone).toBe(true);
    expect(controller.finalize().tolist()).toEqual([[1n, 2n]]);
    expect(streamer.put).toHaveBeenCalledWith([[1n, 2n]]);
    expect(streamer.end).toHaveBeenCalledTimes(1);
  });
});

describe("custom autoregressive sessions", () => {
  it("uses leased CPU logits for arbitrary Transformers.js callbacks", async () => {
    const releases = [jest.fn(), jest.fn()];
    const leases = [createLease([0, 1, 5, 2], releases[0]), createLease([0, 1, 2, 6], releases[1])];
    const session = {
      version: 1,
      batchSize: 1,
      maxSequenceLength: 3,
      prefill: jest.fn(async () => leases[0]),
      decode: jest.fn(async () => leases[1]),
      dispose: jest.fn(async () => {}),
    };
    const backend = {
      modelId: "test/controller-model",
      load: jest.fn(async () => ({
        generation_config: {},
        generationCapabilities: {
          sessionVersion: 1,
          maxBatchSize: 1,
          cpuModes: ["greedy", "multinomial"],
          planModes: [],
          cpuLogits: true,
          declarativePlans: [],
          tokenPipeline: { defaultDepth: 1, maxDepth: 1 },
        },
        createAutoregressiveSession: jest.fn(async () => session),
        async forward(inputs) {
          return inputs;
        },
        async dispose() {},
      })),
    };
    const model = await AutoModel.from_pretrained(backend, {
      config: { model_type: "custom", is_encoder_decoder: false, eos_token_id: 3 },
    });

    const output = await model.generate({
      input_ids: int64Tensor([[1]]),
      attention_mask: int64Tensor([[1]]),
      max_new_tokens: 2,
    });

    expect(output.tolist()).toEqual([[1n, 2n, 3n]]);
    expect(session.prefill).toHaveBeenCalledWith(expect.objectContaining({ inputIds: { data: new Uint32Array([1]), shape: [1, 1] } }));
    expect(session.decode).toHaveBeenCalledWith(expect.objectContaining({ tokenIds: { data: new Uint32Array([2]), shape: [1, 1] } }));
    expect(releases.every((release) => release.mock.calls.length === 1)).toBe(true);
    expect(session.dispose).toHaveBeenCalledTimes(1);
  });

  it("commits runtime plan decisions without reading full logits", async () => {
    const iteratorClosed = jest.fn();
    const session = {
      version: 1,
      batchSize: 1,
      maxSequenceLength: 3,
      prefill: jest.fn(),
      decode: jest.fn(),
      async *generateWithPlan(_inputs, plan) {
        try {
          expect(plan.sampler).toEqual({ op: "argmax" });
          expect(plan.pipelineDepth).toBe(4);
          yield { tokenIds: new Uint32Array([2]) };
          yield { tokenIds: new Uint32Array([3]) };
          yield { tokenIds: new Uint32Array([0]) };
        } finally {
          iteratorClosed();
        }
      },
      dispose: jest.fn(async () => {}),
    };
    const backend = {
      modelId: "test/fast-controller-model",
      load: jest.fn(async () => ({
        generation_config: {},
        generationCapabilities: {
          sessionVersion: 1,
          maxBatchSize: 1,
          cpuModes: ["greedy"],
          planModes: ["greedy"],
          cpuLogits: false,
          declarativePlans: ["argmax"],
          tokenPipeline: { defaultDepth: 4, maxDepth: 4 },
        },
        createAutoregressiveSession: jest.fn(async () => session),
        async forward(inputs) {
          return inputs;
        },
        async dispose() {},
      })),
    };
    const model = await AutoModel.from_pretrained(backend, {
      config: { model_type: "custom", is_encoder_decoder: false, eos_token_id: 3 },
    });

    const output = await model.generate({ input_ids: int64Tensor([[1]]), max_new_tokens: 2 });

    expect(output.tolist()).toEqual([[1n, 2n, 3n]]);
    expect(session.prefill).not.toHaveBeenCalled();
    expect(session.decode).not.toHaveBeenCalled();
    expect(iteratorClosed).toHaveBeenCalledTimes(1);
    expect(session.dispose).toHaveBeenCalledTimes(1);
  });

  it("rejects unsupported batches before creating a runtime session", async () => {
    const createAutoregressiveSession = jest.fn();
    const backend = {
      modelId: "test/batch-controller-model",
      load: jest.fn(async () => ({
        generation_config: {},
        generationCapabilities: {
          sessionVersion: 1,
          maxBatchSize: 1,
          cpuModes: ["greedy"],
          planModes: [],
          cpuLogits: true,
          declarativePlans: [],
          tokenPipeline: { defaultDepth: 1, maxDepth: 1 },
        },
        createAutoregressiveSession,
        async forward(inputs) {
          return inputs;
        },
        async dispose() {},
      })),
    };
    const model = await AutoModel.from_pretrained(backend, {
      config: { model_type: "custom", is_encoder_decoder: false },
    });

    await expect(model.generate({ input_ids: int64Tensor([[1], [2]]), max_new_tokens: 1 })).rejects.toThrow("supports batch size 1");
    expect(createAutoregressiveSession).not.toHaveBeenCalled();
  });
});
