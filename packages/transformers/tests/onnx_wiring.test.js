import { jest } from "@jest/globals";

import { env } from "../src/env.js";
import { getOnnxProviderModule } from "../src/backends/default.js";
import { PreTrainedModel } from "../src/models/modeling_utils.js";

describe("ONNX module wiring", () => {
  it("preserves the ORT environment installed through a pre-registered host", async () => {
    await getOnnxProviderModule();

    expect(typeof env.backends.onnx.setLogLevel).toBe("function");
    expect(env.backends.onnx.wasm).toBeDefined();
  });

  it("honors an aborted signal before built-in model loading", async () => {
    class TestModel extends PreTrainedModel {}
    TestModel._from_pretrained = jest.fn();
    const controller = new AbortController();
    const reason = new Error("cancel loading");
    controller.abort(reason);

    await expect(
      TestModel.from_pretrained("test/aborted-model", {
        config: { model_type: "custom" },
        signal: controller.signal,
      }),
    ).rejects.toBe(reason);
    expect(TestModel._from_pretrained).not.toHaveBeenCalled();
  });
});
