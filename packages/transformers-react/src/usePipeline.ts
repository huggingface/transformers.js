import { useState, useRef, useMemo, useEffect } from "react";
import {
  pipeline,
  type PipelineType,
  type PretrainedModelOptions,
} from "@huggingface/transformers";

type PipelineFn<T extends PipelineType> = Awaited<
  ReturnType<typeof pipeline<T>>
>;

export enum UsePipelineStatus {
  Preload = "preload",
  Loading = "loading",
  Ready = "ready",
}

export interface UsePipelineResult<T extends PipelineType> {
  pipe: PipelineFn<T> | null;
  status: UsePipelineStatus;
}

export function usePipeline<T extends PipelineType>(
  task: T,
  model: string,
  options?: PretrainedModelOptions,
): UsePipelineResult<T> {
  const [status, setStatus] = useState<UsePipelineStatus>(
    UsePipelineStatus.Preload,
  );
  const [pipe, setPipe] = useState<PipelineFn<T> | null>(null);

  const loadingRef = useRef<boolean>(false);

  const optionsKey = useMemo(() => JSON.stringify(options ?? {}), [options]);

  // Load pipeline when task, model, or options change
  useEffect(() => {
    let cancelled = false;

    const loadPipeline = async () => {
      if (loadingRef.current) return;

      loadingRef.current = true;
      setStatus(UsePipelineStatus.Loading);
      setPipe(null);

      try {
        const loadedPipeline = await pipeline(task, model, options);
        if (!cancelled) {
          setPipe(() => loadedPipeline as PipelineFn<T>);
          setStatus(UsePipelineStatus.Ready);
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(UsePipelineStatus.Preload);
          // Don't throw - just set status to Preload
          // Users can check status to see if load failed
        }
      } finally {
        loadingRef.current = false;
      }
    };

    loadPipeline();

    return () => {
      cancelled = true;
    };
  }, [task, model, optionsKey]);

  return {
    pipe,
    status,
  };
}

export default usePipeline;
