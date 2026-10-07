import assert from "node:assert/strict";
import test from "node:test";
import { AutoModelForCausalLM, AutoTokenizer, ModelRegistry } from "@huggingface/transformers";
import { Model } from "../src/Model";

test("forwards the configured revision to loading and cache operations", async () => {
  const tokenizer = AutoTokenizer as unknown as { calls: unknown[][] };
  const causalModel = AutoModelForCausalLM as unknown as { calls: unknown[][] };
  const registry = ModelRegistry as unknown as { calls: unknown[][] };
  tokenizer.calls.length = 0;
  causalModel.calls.length = 0;
  registry.calls.length = 0;

  const model = new Model({
    modelId: "onnx-community/gemma-4-E2B-it-ONNX",
    revision: "refs/pr/5",
    device: "webgpu",
    dtype: "q4f16",
  });
  await model.init();
  await model.isCached();
  assert.equal(await model.downloadSize(), 0);

  assert.equal(model.revision, "refs/pr/5");
  assert.deepEqual(tokenizer.calls, [["onnx-community/gemma-4-E2B-it-ONNX", { revision: "refs/pr/5" }]]);
  assert.deepEqual(causalModel.calls, [
    [
      "onnx-community/gemma-4-E2B-it-ONNX",
      {
        revision: "refs/pr/5",
        device: "webgpu",
        dtype: "q4f16",
        progress_callback: undefined,
      },
    ],
  ]);
  assert.deepEqual(registry.calls, [
    ["is_pipeline_cached", "text-generation", "onnx-community/gemma-4-E2B-it-ONNX", { revision: "refs/pr/5", device: "webgpu", dtype: "q4f16" }],
    ["get_pipeline_files", "text-generation", "onnx-community/gemma-4-E2B-it-ONNX", { revision: "refs/pr/5", device: "webgpu", dtype: "q4f16" }],
    ["get_file_metadata", "onnx-community/gemma-4-E2B-it-ONNX", "model.onnx", { revision: "refs/pr/5" }],
  ]);
});

test("defaults revision to main", () => {
  assert.equal(new Model({ modelId: "test-model" }).revision, "main");
});
