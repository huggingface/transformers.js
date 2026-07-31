import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import webWorkerPipeline from "../src/webWorkerPipeline.js";

const REQUEST = "transformersjs_worker_pipeline";
const RESPONSE_RESULT = "transformersjs_worker_result";
const RESPONSE_CALLBACK_INVOCATION = "callback_bridge:invoke";

class MockWorker {
  postMessage = jest.fn<(message: any) => void>();
  private listeners = new Map<string, Set<(event: any) => void>>();

  addEventListener(type: string, listener: (event: any) => void) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: (event: any) => void) {
    this.listeners.get(type)?.delete(listener);
  }

  dispatch(type: string, data?: any) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(type === "message" ? { data } : (data ?? { type }));
    }
  }

  respond(message: any, result: any = null) {
    this.dispatch("message", { id: message.id, type: RESPONSE_RESULT, result });
  }
}

describe("webWorkerPipeline", () => {
  let worker: MockWorker;

  beforeEach(() => {
    worker = new MockWorker();
  });

  it("initializes and runs a pipeline with falsy inputs", async () => {
    worker.postMessage.mockImplementation((message) => worker.respond(message, message.data));

    const pipe = await webWorkerPipeline<string, string>(worker as unknown as Worker, "text-classification", "test-model");
    await expect(pipe("")).resolves.toBe("");

    expect(worker.postMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        id: 0,
        type: REQUEST,
        operation: "init",
        pipelineId: "pipeline_0",
        task: "text-classification",
        model_id: "test-model",
      }),
    );
    expect(worker.postMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        id: 1,
        operation: "run",
        pipelineId: "pipeline_0",
        data: "",
        pipeOptions: {},
      }),
    );
  });

  it("routes concurrent pipelines through one worker without ID collisions", async () => {
    worker.postMessage.mockImplementation((message) => worker.respond(message, `${message.pipelineId}:${message.data ?? "ready"}`));

    const first = await webWorkerPipeline(worker as unknown as Worker, "text-classification", "first-model");
    const second = await webWorkerPipeline(worker as unknown as Worker, "text-classification", "second-model");

    await expect(Promise.all([first("one"), second("two")])).resolves.toEqual(["pipeline_0:one", "pipeline_1:two"]);
    expect(worker.postMessage.mock.calls.map(([message]) => message.id)).toEqual([0, 1, 2, 3]);
  });

  it("serializes callbacks once and invokes them on the main thread", async () => {
    const callback = jest.fn();
    worker.postMessage.mockImplementation((message) => worker.respond(message));

    const pipe = await webWorkerPipeline(worker as unknown as Worker, "text-classification", "test-model", {
      progress_callback: callback,
    });
    const initMessage = worker.postMessage.mock.calls[0][0];
    const functionId = initMessage.options.progress_callback.functionId;

    worker.dispatch("message", {
      type: RESPONSE_CALLBACK_INVOCATION,
      functionId,
      args: [{ progress: 50 }],
    });
    await pipe("input");

    expect(callback).toHaveBeenCalledWith({ progress: 50 });
    expect(worker.postMessage.mock.calls[1][0]).not.toHaveProperty("options");
  });

  it("rejects serialized worker errors", async () => {
    worker.postMessage.mockImplementation((message) => {
      worker.dispatch("message", {
        id: message.id,
        type: RESPONSE_RESULT,
        error: { name: "RangeError", message: "Model failed" },
      });
    });

    await expect(webWorkerPipeline(worker as unknown as Worker, "text-classification", "test-model")).rejects.toMatchObject({ name: "RangeError", message: "Model failed" });
  });

  it("rejects pending requests when the worker fails", async () => {
    const pipePromise = webWorkerPipeline(worker as unknown as Worker, "text-classification", "test-model");
    worker.dispatch("error", { error: new Error("Worker crashed") });
    await expect(pipePromise).rejects.toThrow("Worker crashed");

    worker.postMessage.mockImplementation((message) => worker.respond(message));
    await expect(webWorkerPipeline(worker as unknown as Worker, "text-classification", "other-model")).resolves.toEqual(expect.any(Function));
  });

  it("contains callback exceptions", async () => {
    const callbackError = new Error("Callback failed");
    const callback = jest.fn(() => {
      throw callbackError;
    });
    const consoleError = jest.spyOn(console, "error").mockImplementation(() => undefined);
    worker.postMessage.mockImplementation((message) => worker.respond(message));
    const pipe = await webWorkerPipeline(worker as unknown as Worker, "text-classification", "test-model", {
      progress_callback: callback,
    });
    const functionId = worker.postMessage.mock.calls[0][0].options.progress_callback.functionId;

    expect(() => worker.dispatch("message", { type: RESPONSE_CALLBACK_INVOCATION, functionId, args: [] })).not.toThrow();
    expect(consoleError).toHaveBeenCalledWith("CallbackBridgeClient: Callback invocation failed", callbackError);

    consoleError.mockRestore();
    await pipe.dispose();
  });

  it("disposes the worker pipeline and prevents later calls", async () => {
    worker.postMessage.mockImplementation((message) => worker.respond(message));
    const pipe = await webWorkerPipeline(worker as unknown as Worker, "text-classification", "test-model");

    const firstDispose = pipe.dispose();
    const secondDispose = pipe.dispose();
    expect(secondDispose).toBe(firstDispose);
    await firstDispose;

    expect(worker.postMessage.mock.calls[1][0]).toEqual(expect.objectContaining({ operation: "dispose", pipelineId: "pipeline_0" }));
    expect(worker.postMessage).toHaveBeenCalledTimes(2);
    await expect(pipe("input")).rejects.toThrow("has been disposed");
  });

  it("allows disposal to be retried after a failed request", async () => {
    let disposeAttempts = 0;
    worker.postMessage.mockImplementation((message) => {
      if (message.operation === "dispose" && disposeAttempts++ === 0) {
        worker.dispatch("message", {
          id: message.id,
          type: RESPONSE_RESULT,
          error: { name: "Error", message: "Temporary disposal failure" },
        });
      } else {
        worker.respond(message);
      }
    });
    const pipe = await webWorkerPipeline(worker as unknown as Worker, "text-classification", "test-model");

    await expect(pipe.dispose()).rejects.toThrow("Temporary disposal failure");
    await expect(pipe.dispose()).resolves.toBeUndefined();
    await expect(pipe("input")).rejects.toThrow("has been disposed");
    expect(disposeAttempts).toBe(2);
  });
});
