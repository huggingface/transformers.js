import { renderHook, waitFor, act } from "@testing-library/react";
import { usePipeline, UsePipelineStatus } from "../usePipeline";
import { pipeline } from "@huggingface/transformers";

// Mock the transformers pipeline
jest.mock("@huggingface/transformers", () => ({
  pipeline: jest.fn(),
}));

const mockPipeline = pipeline as jest.MockedFunction<typeof pipeline>;

describe("usePipeline", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should automatically load pipeline on mount", async () => {
    const mockPipelineFn = jest.fn().mockResolvedValue({ label: "positive" });
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result } = renderHook(() =>
      usePipeline("text-classification", "test-model"),
    );

    // Status should be Preload or Loading depending on timing

    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    expect(result.current.pipe).toBe(mockPipelineFn);
    expect(mockPipeline).toHaveBeenCalledWith(
      "text-classification",
      "test-model",
      undefined,
    );
  });

  it("should pass options to pipeline", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result } = renderHook(() =>
      usePipeline("text-classification", "test-model", {
        device: "cpu",
      } as any),
    );

    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledWith(
      "text-classification",
      "test-model",
      { device: "cpu" },
    );
  });

  it("should reload when task changes", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result, rerender } = renderHook(
      ({ task, model }: { task: any; model: string }) =>
        usePipeline(task, model),
      {
        initialProps: { task: "text-classification", model: "model1" },
      },
    );

    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledTimes(1);

    // Change task
    act(() => {
      rerender({ task: "summarization", model: "model1" });
    });

    await waitFor(() => {
      expect(mockPipeline).toHaveBeenCalledTimes(2);
    });

    expect(mockPipeline).toHaveBeenLastCalledWith(
      "summarization",
      "model1",
      undefined,
    );
  });

  it("should reload when model changes", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result, rerender } = renderHook(
      ({ model }) => usePipeline("text-classification", model),
      {
        initialProps: { model: "model1" },
      },
    );

    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledTimes(1);

    // Change model
    act(() => {
      rerender({ model: "model2" });
    });

    await waitFor(() => {
      expect(mockPipeline).toHaveBeenCalledTimes(2);
    });

    expect(mockPipeline).toHaveBeenLastCalledWith(
      "text-classification",
      "model2",
      undefined,
    );
  });

  it("should reload when options change", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result, rerender } = renderHook(
      ({ options }: { options: any }) =>
        usePipeline("text-classification", "model", options),
      {
        initialProps: { options: { device: "cpu" } },
      },
    );

    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    expect(mockPipeline).toHaveBeenCalledTimes(1);

    // Change options
    act(() => {
      rerender({ options: { device: "gpu" } });
    });

    await waitFor(() => {
      expect(mockPipeline).toHaveBeenCalledTimes(2);
    });

    expect(mockPipeline).toHaveBeenLastCalledWith(
      "text-classification",
      "model",
      { device: "gpu" },
    );
  });

  it("should handle loading errors", async () => {
    const error = new Error("Failed to load");
    mockPipeline.mockRejectedValue(error);

    const { result } = renderHook(() =>
      usePipeline("text-classification", "test-model"),
    );

    // Wait a bit for the error to be processed
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100));
    });

    expect(result.current.pipe).toBeNull();
  });

  it("should set loading status while loading", async () => {
    let resolveLoad: (value: any) => void;
    const loadPromise = new Promise<any>((resolve) => {
      resolveLoad = resolve;
    });
    mockPipeline.mockReturnValue(loadPromise as any);

    const { result } = renderHook(() =>
      usePipeline("text-classification", "test-model"),
    );

    // After a tick, should be Loading
    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Loading);
    });

    // Resolve the promise
    const mockPipelineFn = jest.fn();
    act(() => {
      resolveLoad!(mockPipelineFn);
    });

    // Should become Ready
    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    expect(result.current.pipe).toBe(mockPipelineFn);
  });

  it("should cleanup on unmount", async () => {
    const mockPipelineFn = jest.fn();
    mockPipeline.mockResolvedValue(mockPipelineFn as any);

    const { result, unmount } = renderHook(() =>
      usePipeline("text-classification", "test-model"),
    );

    await waitFor(() => {
      expect(result.current.status).toBe(UsePipelineStatus.Ready);
    });

    act(() => {
      unmount();
    });

    // Should not throw or cause any issues
    expect(result.current.pipe).toBe(mockPipelineFn);
  });
});
