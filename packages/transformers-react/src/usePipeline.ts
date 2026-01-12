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
  progress: number;
}

interface ProgressData {
  status?: string;
  name?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

export function usePipeline<T extends PipelineType>(
  task: T,
  model: string,
  options?: PretrainedModelOptions,
): UsePipelineResult<T> {
  const [status, setStatus] = useState<UsePipelineStatus>(
    UsePipelineStatus.Preload,
  );
  const [progress, setProgress] = useState<number>(0);
  const [pipe, setPipe] = useState<PipelineFn<T> | null>(null);

  const loadingRef = useRef<boolean>(false);
  const progressElements = useRef<
    Record<string, { loaded: number; total: number }>
  >({});

  const optionsKey = useMemo(() => JSON.stringify(options ?? {}), [options]);

  // Load pipeline when task, model, or options change
  useEffect(() => {
    let cancelled = false;

    const loadPipeline = async () => {
      if (loadingRef.current) return;

      loadingRef.current = true;
      progressElements.current = {};
      setStatus(UsePipelineStatus.Loading);
      setProgress(0);
      setPipe(null);

      const pipelineOptions: PretrainedModelOptions = {
        ...options,
        progress_callback: (progressData: ProgressData) => {
          // Call user's progress callback if provided
          if (options?.progress_callback) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (options.progress_callback as (data: ProgressData) => void)(
              progressData,
            );
          }

          // Track loading progress
          if (progressData.status === "progress" && progressData.file) {
            progressElements.current[progressData.file] = {
              loaded: progressData.loaded ?? 0,
              total: progressData.total ?? 0,
            };

            let allTotal = 0;
            let allLoaded = 0;
            let hasOnnxFile = false;

            for (const [file, { loaded, total }] of Object.entries(
              progressElements.current,
            )) {
              allTotal += total;
              allLoaded += loaded;
              if (file.endsWith(".onnx")) {
                hasOnnxFile = true;
              }
            }

            if (hasOnnxFile && allTotal > 0) {
              setProgress(Math.round((allLoaded / allTotal) * 100));
            }
          }
        },
      };

      try {
        const loadedPipeline = await pipeline(task, model, pipelineOptions);
        if (!cancelled) {
          setPipe(() => loadedPipeline as PipelineFn<T>);
          setStatus(UsePipelineStatus.Ready);
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(UsePipelineStatus.Preload);
          throw error;
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
    progress,
  };
}

export default usePipeline;
