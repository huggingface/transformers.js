import { describe, it, expect, jest, beforeEach, afterAll } from "@jest/globals";

const pipelineMock = jest.fn<any>();
jest.unstable_mockModule("@huggingface/transformers", () => ({ pipeline: pipelineMock }));

const { default: webWorkerPipelineHandler } = await import("../src/webWorkerPipelineHandler.js");

const REQUEST = "transformersjs_worker_pipeline";
const RESPONSE_RESULT = "transformersjs_worker_result";
const RESPONSE_CALLBACK_INVOCATION = "callback_bridge:invoke";

const request = (data: Record<string, any>) => ({ data: { id: 0, type: REQUEST, pipelineId: "pipeline_0", ...data } }) as MessageEvent;

describe("webWorkerPipelineHandler", () => {
  let postMessage: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    postMessage = jest.fn();
    (globalThis as any).self = { postMessage };
  });

  afterAll(() => {
    delete (globalThis as any).self;
  });

  it("ignores unrelated messages", async () => {
    const handler = webWorkerPipelineHandler();
    await handler.onmessage({ data: { type: "other" } } as MessageEvent);
    expect(postMessage).not.toHaveBeenCalled();
  });

  it("initializes once and runs falsy inputs", async () => {
    const pipe = Object.assign(jest.fn<any>().mockResolvedValue("result"), { dispose: jest.fn() });
    pipelineMock.mockResolvedValue(pipe);
    const handler = webWorkerPipelineHandler();

    await handler.onmessage(
      request({
        operation: "init",
        task: "text-classification",
        model_id: "test-model",
        options: { device: "cpu" },
      }),
    );
    await handler.onmessage(request({ id: 1, operation: "run", data: "", pipeOptions: { top_k: 1 } }));

    expect(pipelineMock).toHaveBeenCalledTimes(1);
    expect(pipe).toHaveBeenCalledWith("", { top_k: 1 });
    expect(postMessage).toHaveBeenLastCalledWith({ id: 1, type: RESPONSE_RESULT, result: "result" });
  });

  it("bridges callback invocations", async () => {
    pipelineMock.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({ progress: 50 });
      return Object.assign(jest.fn(), { dispose: jest.fn() });
    });
    const handler = webWorkerPipelineHandler();

    await handler.onmessage(
      request({
        operation: "init",
        task: "text-classification",
        model_id: "test-model",
        options: { progress_callback: { __fn: true, functionId: "cb_0" } },
      }),
    );

    expect(postMessage).toHaveBeenNthCalledWith(1, {
      type: RESPONSE_CALLBACK_INVOCATION,
      functionId: "cb_0",
      args: [{ progress: 50 }],
    });
  });

  it("returns structured initialization and execution errors", async () => {
    pipelineMock.mockRejectedValue(new Error("Could not load model"));
    const handler = webWorkerPipelineHandler();

    await handler.onmessage(request({ operation: "init", task: "text-classification", model_id: "test-model", options: {} }));

    expect(postMessage).toHaveBeenCalledWith({
      id: 0,
      type: RESPONSE_RESULT,
      error: expect.objectContaining({ name: "Error", message: "Could not load model" }),
    });
  });

  it("rejects duplicate pipeline IDs without leaking the original", async () => {
    const originalPipe = Object.assign(jest.fn(), { dispose: jest.fn() });
    pipelineMock.mockResolvedValue(originalPipe);
    const handler = webWorkerPipelineHandler();
    const initRequest = request({ operation: "init", task: "text-classification", model_id: "test-model", options: {} });

    await handler.onmessage(initRequest);
    await handler.onmessage(initRequest);

    expect(pipelineMock).toHaveBeenCalledTimes(1);
    expect(originalPipe.dispose).not.toHaveBeenCalled();
    expect(postMessage).toHaveBeenLastCalledWith({
      id: 0,
      type: RESPONSE_RESULT,
      error: expect.objectContaining({ message: "Web Worker pipeline already exists: pipeline_0" }),
    });
  });

  it("disposes and removes initialized pipelines", async () => {
    const dispose = jest.fn<() => Promise<void>>().mockResolvedValue();
    pipelineMock.mockResolvedValue(Object.assign(jest.fn(), { dispose }));
    const handler = webWorkerPipelineHandler();

    await handler.onmessage(request({ operation: "init", task: "text-classification", model_id: "test-model", options: {} }));
    await handler.onmessage(request({ id: 1, operation: "dispose" }));
    await handler.onmessage(request({ id: 2, operation: "run", data: "input" }));

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(postMessage).toHaveBeenLastCalledWith({
      id: 2,
      type: RESPONSE_RESULT,
      error: expect.objectContaining({ message: "Unknown Web Worker pipeline: pipeline_0" }),
    });
  });

  it("waits for in-flight inference before disposing a pipeline", async () => {
    let finishRun!: () => void;
    let markRunStarted!: () => void;
    const runStarted = new Promise<void>((resolve) => (markRunStarted = resolve));
    const pipe = Object.assign(
      jest.fn<any>().mockImplementation(
        () =>
          new Promise<void>((resolve) => {
            finishRun = resolve;
            markRunStarted();
          }),
      ),
      { dispose: jest.fn<() => Promise<void>>().mockResolvedValue() },
    );
    pipelineMock.mockResolvedValue(pipe);
    const handler = webWorkerPipelineHandler();
    await handler.onmessage(request({ operation: "init", task: "text-classification", model_id: "test-model", options: {} }));

    const runPromise = handler.onmessage(request({ id: 1, operation: "run", data: "input" }));
    const disposePromise = handler.onmessage(request({ id: 2, operation: "dispose" }));
    await runStarted;
    expect(pipe.dispose).not.toHaveBeenCalled();

    finishRun();
    await Promise.all([runPromise, disposePromise]);
    expect(pipe.dispose).toHaveBeenCalledTimes(1);
  });

  it("keeps a pipeline available when disposal fails", async () => {
    const dispose = jest.fn<() => Promise<void>>().mockRejectedValueOnce(new Error("Temporary failure")).mockResolvedValue();
    pipelineMock.mockResolvedValue(Object.assign(jest.fn(), { dispose }));
    const handler = webWorkerPipelineHandler();
    await handler.onmessage(request({ operation: "init", task: "text-classification", model_id: "test-model", options: {} }));

    await handler.onmessage(request({ id: 1, operation: "dispose" }));
    await handler.onmessage(request({ id: 2, operation: "dispose" }));

    expect(dispose).toHaveBeenCalledTimes(2);
    expect(postMessage).toHaveBeenNthCalledWith(2, expect.objectContaining({ id: 1, error: expect.objectContaining({ message: "Temporary failure" }) }));
    expect(postMessage).toHaveBeenLastCalledWith({ id: 2, type: RESPONSE_RESULT, result: null });
  });
});
