import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { PipelineType } from "@huggingface/transformers";

const REQUEST_MESSAGE_TYPE = "transformersjs_worker_pipeline";
const RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK = "transformersjs_worker_invokeCallback";
const RESPONSE_MESSAGE_TYPE_RESULT = "transformersjs_worker_result";

type WebWorkerPipeline = <PayloadType = any, ResultType = any>(worker: Worker, task: PipelineType, model_id: string, options?: Record<string, any>) => Promise<(data: PayloadType, pipeOptions?: Record<string, any>) => Promise<ResultType>>;

interface MockWorker {
  postMessage: jest.Mock;
  onmessage: ((e: MessageEvent) => void) | null;
}

describe("webWorkerPipeline", () => {
  let webWorkerPipeline: WebWorkerPipeline;
  let mockWorker: MockWorker;

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
      mockWorker.onmessage?.({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      } as MessageEvent);
    }, 0);

    const pipeline = await webWorkerPipeline(mockWorker as any, "sentiment-analysis", "test-model");

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
    const task: PipelineType = "text-classification";
    const modelId = "my-model";
    const options = { device: "cpu" };

    setTimeout(() => {
      mockWorker.onmessage?.({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      } as MessageEvent);
    }, 0);

    await webWorkerPipeline(mockWorker as any, task, modelId, options);

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
      mockWorker.onmessage?.({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      } as MessageEvent);
    }, 0);

    await webWorkerPipeline(mockWorker as any, "text-classification", "test-model", options);

    const callArgs = mockWorker.postMessage.mock.calls[0][0] as any;
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
      mockWorker.onmessage?.({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      } as MessageEvent);

      // Then simulate callback invocation
      setTimeout(() => {
        mockWorker.onmessage?.({
          data: {
            type: RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
            functionId: "cb_progress_callback",
            args: [{ status: "progress", progress: 50 }],
          },
        } as MessageEvent);
      }, 10);
    }, 0);

    await webWorkerPipeline(mockWorker as any, "text-classification", "test-model", options);

    // Wait for callback to be invoked
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(callback).toHaveBeenCalledWith({ status: "progress", progress: 50 });
  });

  it("should send pipeline execution messages", async () => {
    setTimeout(() => {
      mockWorker.onmessage?.({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      } as MessageEvent);
    }, 0);

    const pipeline = await webWorkerPipeline(mockWorker as any, "sentiment-analysis", "test-model");

    // Reset mock to clear init call
    mockWorker.postMessage.mockClear();

    // Simulate execution
    const executePromise = pipeline("I love this!", { top_k: 1 });

    // Simulate worker response
    setTimeout(() => {
      mockWorker.onmessage?.({
        data: {
          id: 0,
          type: RESPONSE_MESSAGE_TYPE_RESULT,
          result: [{ label: "POSITIVE", score: 0.99 }],
        },
      } as MessageEvent);
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
      mockWorker.onmessage?.({
        data: { id: "init", type: RESPONSE_MESSAGE_TYPE_RESULT },
      } as MessageEvent);
    }, 0);

    const pipeline = await webWorkerPipeline(mockWorker as any, "text-classification", "test-model");

    const executePromise = pipeline("test input");

    // Simulate worker error
    setTimeout(() => {
      mockWorker.onmessage?.({
        data: {
          id: 0,
          type: RESPONSE_MESSAGE_TYPE_RESULT,
          error: new Error("Pipeline execution failed"),
        },
      } as MessageEvent);
    }, 10);

    await expect(executePromise).rejects.toThrow("Pipeline execution failed");
  });
});
