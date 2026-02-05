import { describe, it, expect, jest, beforeEach } from "@jest/globals";

// Import constants to match the implementation
const REQUEST_MESSAGE_TYPE = "transformersjs_worker_pipeline";
const RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK = "transformersjs_worker_invokeCallback";
const RESPONSE_MESSAGE_TYPE_RESULT = "transformersjs_worker_result";

describe("webWorkerPipeline", () => {
  let webWorkerPipeline;
  let mockWorker;

  beforeEach(async () => {
    // Import the built module
    const module = await import("../dist/index.js");
    webWorkerPipeline = module.webWorkerPipeline;

    // Create a mock Worker
    mockWorker = {
      postMessage: jest.fn(),
      onmessage: null,
    };
  });

  it("should create a pipeline function", async () => {
    // Setup the worker to respond
    setTimeout(() => {
      mockWorker.onmessage({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      });
    }, 0);

    const pipeline = await webWorkerPipeline(mockWorker, "sentiment-analysis", "test-model");

    expect(typeof pipeline).toBe("function");
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "init",
        type: REQUEST_MESSAGE_TYPE,
        task: "sentiment-analysis",
        model_id: "test-model",
      }),
    );
  });

  it("should send initialization message with correct parameters", async () => {
    const task = "text-classification";
    const modelId = "my-model";
    const options = { device: "cpu" };

    setTimeout(() => {
      mockWorker.onmessage({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      });
    }, 0);

    await webWorkerPipeline(mockWorker, task, modelId, options);

    expect(mockWorker.postMessage).toHaveBeenCalledWith({
      id: "init",
      type: REQUEST_MESSAGE_TYPE,
      data: null,
      task: task,
      model_id: modelId,
      options: options,
    });
  });

  it("should serialize function options", async () => {
    const callback = jest.fn();
    const options = {
      progress_callback: callback,
      device: "cpu",
    };

    setTimeout(() => {
      mockWorker.onmessage({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      });
    }, 0);

    await webWorkerPipeline(mockWorker, "test-task", "test-model", options);

    const callArgs = mockWorker.postMessage.mock.calls[0][0];
    expect(callArgs.options.progress_callback).toEqual({
      __fn: true,
      functionId: "cb_progress_callback",
    });
    expect(callArgs.options.device).toBe("cpu");
  });

  it("should handle callback invocations from worker", async () => {
    const callback = jest.fn();
    const options = {
      progress_callback: callback,
    };

    setTimeout(() => {
      // First, send init response
      mockWorker.onmessage({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      });

      // Then simulate callback invocation
      setTimeout(() => {
        mockWorker.onmessage({
          data: {
            type: RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
            functionId: "cb_progress_callback",
            args: [{ status: "progress", progress: 50 }],
          },
        });
      }, 10);
    }, 0);

    await webWorkerPipeline(mockWorker, "test-task", "test-model", options);

    // Wait for callback to be invoked
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(callback).toHaveBeenCalledWith({ status: "progress", progress: 50 });
  });

  it("should send pipeline execution messages", async () => {
    setTimeout(() => {
      mockWorker.onmessage({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      });
    }, 0);

    const pipeline = await webWorkerPipeline(mockWorker, "sentiment-analysis", "test-model");

    // Reset mock to clear init call
    mockWorker.postMessage.mockClear();

    // Simulate execution
    const executePromise = pipeline("I love this!", { top_k: 1 });

    // Simulate worker response
    setTimeout(() => {
      mockWorker.onmessage({
        data: {
          id: 0,
          type: RESPONSE_MESSAGE_TYPE_RESULT,
          result: [{ label: "POSITIVE", score: 0.99 }],
        },
      });
    }, 10);

    const result = await executePromise;

    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 0,
        type: REQUEST_MESSAGE_TYPE,
        data: "I love this!",
        task: "sentiment-analysis",
        model_id: "test-model",
        pipeOptions: { top_k: 1 },
      }),
    );

    expect(result).toEqual([{ label: "POSITIVE", score: 0.99 }]);
  });

  it("should handle errors from worker", async () => {
    setTimeout(() => {
      mockWorker.onmessage({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      });
    }, 0);

    const pipeline = await webWorkerPipeline(mockWorker, "test-task", "test-model");

    const executePromise = pipeline("test input");

    // Simulate worker error
    setTimeout(() => {
      mockWorker.onmessage({
        data: {
          id: 0,
          type: RESPONSE_MESSAGE_TYPE_RESULT,
          error: new Error("Pipeline execution failed"),
        },
      });
    }, 10);

    await expect(executePromise).rejects.toThrow("Pipeline execution failed");
  });
});
