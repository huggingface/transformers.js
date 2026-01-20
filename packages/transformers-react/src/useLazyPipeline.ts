import { useState, useRef, useMemo, useCallback } from "react";
import {
  pipeline,
  type PipelineType,
  type PretrainedModelOptions,
} from "@huggingface/transformers";

type PipelineFn<T extends PipelineType> = Awaited<
  ReturnType<typeof pipeline<T>>
>;

export enum UseLazyPipelineStatus {
  Idle = "idle",
  Loading = "loading",
  Ready = "ready",
  Error = "error",
}

export interface UseLazyPipelineResult<T extends PipelineType> {
  pipe: PipelineFn<T> | null;
  status: UseLazyPipelineStatus;
  load: () => Promise<void>;
  error: Error | null;
  run: PipelineFn<T>;
}

export function useLazyPipeline<T extends PipelineType>(
  task: T,
  model: string,
  options?: PretrainedModelOptions,
): UseLazyPipelineResult<T> {
  const [status, setStatus] = useState<UseLazyPipelineStatus>(
    UseLazyPipelineStatus.Idle,
  );
  const [pipe, setPipe] = useState<PipelineFn<T> | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const loadingRef = useRef<boolean>(false);
  const loadedConfigRef = useRef<string | null>(null);

  const optionsKey = useMemo(() => JSON.stringify(options ?? {}), [options]);

  const load = useCallback(async () => {
    const currentConfig = JSON.stringify({
      task,
      model,
      options: optionsKey,
    });

    // Skip loading if the exact same configuration is already loaded
    if (
      loadedConfigRef.current === currentConfig &&
      status === UseLazyPipelineStatus.Ready
    ) {
      return;
    }

    if (loadingRef.current) return;

    loadingRef.current = true;
    setStatus(UseLazyPipelineStatus.Loading);
    setPipe(null);
    setError(null);

    try {
      const loadedPipeline = await pipeline(task, model, options);
      setPipe(() => loadedPipeline as PipelineFn<T>);
      setStatus(UseLazyPipelineStatus.Ready);
      loadedConfigRef.current = currentConfig;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      setStatus(UseLazyPipelineStatus.Error);
      loadedConfigRef.current = null;
      throw error;
    } finally {
      loadingRef.current = false;
    }
  }, [task, model, optionsKey, status, options]);

  const run = useCallback(
    ((...args: any[]) => {
      if (!pipe) {
        throw new Error(
          `Pipeline not loaded. Call load() before using the pipeline. ` +
            `Current status: ${status}. ` +
            `Make sure to call load() and wait for status to be 'ready' before calling run().`,
        );
      }
      return (pipe as any)(...args);
    }) as PipelineFn<T>,
    [pipe, status],
  );

  return {
    pipe,
    status,
    load,
    error,
    run,
  };
}

export default useLazyPipeline;
