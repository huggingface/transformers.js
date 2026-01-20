import { renderHook, act, waitFor } from "@testing-library/react";
import { useLazyPipeline, UseLazyPipelineStatus } from "../useLazyPipeline";
import { pipeline } from "@huggingface/transformers";

// Mock the transformers pipeline
jest.mock("@huggingface/transformers", () => ({
  pipeline: jest.fn(),
}));

const mockPipeline = pipeline as jest.MockedFunction<typeof pipeline>;

describe("useLazyPipeline", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should start in idle state", () => {
    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    expect(result.current.status).toBe(UseLazyPipelineStatus.Idle);
    expect(result.current.pipe).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("should load pipeline when load() is called", async () => {
    const mockPipelineFn = jest.fn().mockResolvedValue({ label: "positive" });
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    expect(result.current.status).toBe(UseLazyPipelineStatus.Idle);

    await act(async () => {
      await result.current.load();
    });

    await waitFor(() => {
      expect(result.current.status).toBe(UseLazyPipelineStatus.Ready);
    });

    expect(result.current.pipe).toBe(mockPipelineFn);
    expect(result.current.error).toBeNull();
    expect(mockPipeline).toHaveBeenCalledWith(
      "text-classification",
      "test-model",
      undefined,
    );
  });

  it("should pass options to pipeline when loading", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model", { device: "cpu" }),
    );

    await act(async () => {
      await result.current.load();
    });

    await waitFor(() => {
      expect(result.current.status).toBe(UseLazyPipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledWith(
      "text-classification",
      "test-model",
      { device: "cpu" },
    );
  });

  it("should pass options including progress_callback to pipeline", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);
    const progressCallback = jest.fn();

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model", {
        device: "cpu",
        progress_callback: progressCallback,
      } as any),
    );

    await act(async () => {
      await result.current.load();
    });

    await waitFor(() => {
      expect(result.current.status).toBe(UseLazyPipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledWith(
      "text-classification",
      "test-model",
      {
        device: "cpu",
        progress_callback: progressCallback,
      },
    );
  });

  it("should not reload if same configuration is already loaded", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    await act(async () => {
      await result.current.load();
    });

    await waitFor(() => {
      expect(result.current.status).toBe(UseLazyPipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledTimes(1);

    // Call load again with same config
    await act(async () => {
      await result.current.load();
    });

    // Should not call pipeline again
    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it("should handle loading errors", async () => {
    const error = new Error("Failed to load");
    mockPipeline.mockRejectedValue(error);

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    await act(async () => {
      try {
        await result.current.load();
      } catch (e) {
        // Expected to throw
      }
    });

    await waitFor(() => {
      expect(result.current.status).toBe(UseLazyPipelineStatus.Error);
    });

    expect(result.current.error).toEqual(error);
    expect(result.current.pipe).toBeNull();
  });

  it("should throw descriptive error when run() is called before loading", () => {
    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    expect(() => {
      result.current.run("test");
    }).toThrow("Pipeline not loaded");
    expect(() => {
      result.current.run("test");
    }).toThrow("Call load() before using the pipeline");
  });

  it("should run pipeline after loading", async () => {
    const mockPipelineFn = jest.fn().mockResolvedValue({ label: "positive" });
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    await act(async () => {
      await result.current.load();
    });

    await waitFor(() => {
      expect(result.current.status).toBe(UseLazyPipelineStatus.Ready);
    });

    const output = await result.current.run("This is great!");

    expect(mockPipelineFn).toHaveBeenCalledWith("This is great!");
    expect(output).toEqual({ label: "positive" });
  });

  it("should prevent concurrent loads", async () => {
    let resolveLoad: () => void;
    const loadPromise = new Promise<any>((resolve) => {
      resolveLoad = () => resolve(jest.fn());
    });
    mockPipeline.mockReturnValue(loadPromise);

    const { result } = renderHook(() =>
      useLazyPipeline("text-classification", "test-model"),
    );

    // Start first load
    const load1 = act(async () => {
      await result.current.load();
    });

    // Try to start second load immediately
    const load2 = act(async () => {
      await result.current.load();
    });

    // Resolve the mock
    resolveLoad!();

    await Promise.all([load1, load2]);

    // Pipeline should only be called once
    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });
});
