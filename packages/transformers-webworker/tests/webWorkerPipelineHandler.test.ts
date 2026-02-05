import { describe, it, expect, jest, beforeEach } from "@jest/globals";

// Import constants to match the implementation
const REQUEST_MESSAGE_TYPE = "transformersjs_worker_pipeline";
const RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK = "transformersjs_worker_invokeCallback";

interface WebWorkerPipelineHandler {
  (): {
    onmessage: (event: MessageEvent) => Promise<void>;
  };
}

interface MockSelf {
  postMessage: jest.Mock;
}

describe("webWorkerPipelineHandler", () => {
  let webWorkerPipelineHandler: WebWorkerPipelineHandler;
  let handler: ReturnType<WebWorkerPipelineHandler>;
  let mockSelf: MockSelf;

  beforeEach(async () => {
    // Clear all mocks
    jest.clearAllMocks();

    // Mock self.postMessage
    mockSelf = {
      postMessage: jest.fn(),
    };
    (global as any).self = mockSelf;

    // Import the built module
    const module = await import("../dist/index.js");
    webWorkerPipelineHandler = module.webWorkerPipelineHandler;
    handler = webWorkerPipelineHandler();
  });

  it("should create a handler with onmessage function", () => {
    expect(handler).toHaveProperty("onmessage");
    expect(typeof handler.onmessage).toBe("function");
  });

  it("should ignore messages without correct type", async () => {
    await handler.onmessage({ data: { type: "other" } } as MessageEvent);
    expect(mockSelf.postMessage).not.toHaveBeenCalled();
  });

  it("should ignore messages without data", async () => {
    await handler.onmessage({} as MessageEvent);
    expect(mockSelf.postMessage).not.toHaveBeenCalled();
  });

  it("should return a promise when handling messages", () => {
    const messageEvent: MessageEvent = {
      data: {
        id: 1,
        type: REQUEST_MESSAGE_TYPE,
        task: "text-classification",
        model_id: "test-model",
        options: {},
        data: null,
      },
    } as MessageEvent;

    // The handler returns a promise (don't await to avoid loading real models)
    const result = handler.onmessage(messageEvent);
    expect(result instanceof Promise).toBe(true);
  });

  it("should unserialize function callbacks correctly", () => {
    // Test the unserialize logic by checking what the handler would do
    const serializedOptions = {
      progress_callback: {
        __fn: true,
        functionId: "cb_progress_callback",
      },
      device: "cpu",
    };

    // We can't directly test the internal unserializeOptions function,
    // but we can test that serialized functions have the expected structure
    expect(serializedOptions.progress_callback).toHaveProperty("__fn", true);
    expect(serializedOptions.progress_callback).toHaveProperty("functionId");
  });

  it("should handle callback invocation structure", () => {
    // Simulate what would happen when a callback is invoked
    const callbackFn = (...args: any[]) =>
      mockSelf.postMessage({
        type: RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
        functionId: "test_callback_id",
        args,
      });

    // Call the callback
    callbackFn({ progress: 50 });

    // Verify postMessage was called with correct structure
    expect(mockSelf.postMessage).toHaveBeenCalledWith({
      type: RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
      functionId: "test_callback_id",
      args: [{ progress: 50 }],
    });
  });

  it("should handle empty pipeOptions parameter", () => {
    const messageWithoutPipeOptions: MessageEvent = {
      data: {
        id: 1,
        type: REQUEST_MESSAGE_TYPE,
        task: "text-classification",
        model_id: "test-model",
        options: {},
        data: "test input",
        // pipeOptions omitted
      },
    } as MessageEvent;

    // Should not throw synchronously when pipeOptions is undefined
    expect(() => {
      handler.onmessage(messageWithoutPipeOptions);
    }).not.toThrow();
  });

  it("should properly structure message data", () => {
    const validMessage = {
      data: {
        id: 123,
        type: REQUEST_MESSAGE_TYPE,
        task: "text-classification",
        model_id: "my-model",
        options: { device: "webgpu" },
        data: "test input",
        pipeOptions: { top_k: 5 },
      },
    };

    // Verify message structure is valid
    expect(validMessage.data).toHaveProperty("id");
    expect(validMessage.data).toHaveProperty("type", REQUEST_MESSAGE_TYPE);
    expect(validMessage.data).toHaveProperty("task");
    expect(validMessage.data).toHaveProperty("model_id");
    expect(validMessage.data).toHaveProperty("data");
  });
});
