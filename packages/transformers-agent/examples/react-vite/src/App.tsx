import { useEffect, useRef, useState } from "react";
import { Agent, Model } from "@huggingface/transformers-agent";

type Status = "idle" | "working" | "ready" | "error";
type ToolResultView = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  output: unknown;
  durationMs: number;
};

type RoundView = {
  thinkingText: string;
  text: string;
  tools: ToolResultView[];
  usage?: { totalTokens?: number };
};

type RequestResultView = {
  done: boolean;
  runs: RoundView[];
  usage?: { totalTokens?: number };
};

type ResponseView = {
  id: number;
  prompt: string;
  text: string;
  result: RequestResultView | null;
  done: boolean;
  error?: string;
};

const getWeatherTool = {
  description: "Get the weather in a location",
  inputSchema: {
    type: "object" as const,
    properties: {
      location: {
        type: "string" as const,
        description: "The location to get the weather for",
      },
    },
    required: ["location"],
  },
  execute: async (args: Record<string, unknown>) => {
    const location =
      typeof args.location === "string" && args.location.trim().length > 0
        ? args.location.trim()
        : "Unknown";

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            location,
            condition: "Sunny",
            temperatureC: 21,
            source: "mock",
          }),
        },
      ],
    };
  },
};

export function App() {
  const [modelId, setModelId] = useState("onnx-community/Qwen3.5-4B-ONNX-OPT");
  const [prompt, setPrompt] = useState("Whats the weather in Bern?");
  const [status, setStatus] = useState<Status>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [cached, setCached] = useState<boolean | null>(null);
  const [downloadBytes, setDownloadBytes] = useState<number | null>(null);
  const [cachedBytes, setCachedBytes] = useState<number | null>(null);
  const [responses, setResponses] = useState<ResponseView[]>([]);

  const modelRef = useRef<Model>(new Model({ modelId }));
  const agentRef = useRef<Agent | null>(null);
  const responseIdRef = useRef(0);

  useEffect(() => {
    modelRef.current = new Model({ modelId, dtype: "q4" });
    agentRef.current = null;
    setCached(null);
    setDownloadBytes(null);
    setCachedBytes(null);
    setResponses([]);
    setStatus("idle");
  }, [modelId]);

  const addLog = (line: string) =>
    setLog((prev) => [line, ...prev].slice(0, 20));

  const checkCache = async () => {
    setStatus("working");
    try {
      const [isCached, need, have] = await Promise.all([
        modelRef.current.isCached(),
        modelRef.current.downloadSize(),
        modelRef.current.cachedSize(),
      ]);
      setCached(isCached);
      setDownloadBytes(need);
      setCachedBytes(have);
      addLog(
        `Cache checked: cached=${String(isCached)}, missing=${formatBytes(need)}`,
      );
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      addLog(errorMessage(error));
    }
  };

  const initModel = async () => {
    setStatus("working");
    try {
      let progress = 0;
      await modelRef.current.init((info) => {
        if (info.status === "progress_total") {
          const newProgress = Math.round(info.progress);
          if (newProgress !== progress) {
            addLog(`progress: ${newProgress}%`);
            progress = newProgress;
          }
        }
      });
      setStatus("ready");
      addLog("Model initialized.");
    } catch (error) {
      setStatus("error");
      addLog(errorMessage(error));
    }
  };

  const runAgent = async () => {
    setStatus("working");
    const currentPrompt = prompt;
    const responseId = responseIdRef.current + 1;
    responseIdRef.current = responseId;
    setResponses((prev) => [
      ...prev,
      {
        id: responseId,
        prompt: currentPrompt,
        text: "",
        result: null,
        done: false,
      },
    ]);
    try {
      if (!agentRef.current) {
        agentRef.current = new Agent({
          model: modelRef.current,
          system:
            "You are a concise assistant. After you call a tool, always answer the question.",
          tools: {
            getWeather: getWeatherTool,
          },
          enableThinking: false,
        });
      }
      let finalOutput: RequestResultView | null = null;
      for await (const chunk of agentRef.current.stream(currentPrompt)) {
        const typedChunk = chunk as unknown as RequestResultView;
        const text = typedChunk.runs.map((round) => round.text).join("");
        setResponses((prev) =>
          prev.map((response) =>
            response.id === responseId
              ? {
                  ...response,
                  text,
                  result: typedChunk,
                  done: typedChunk.done,
                }
              : response,
          ),
        );
        //console.log(typedChunk);
        finalOutput = typedChunk;
      }

      if (!finalOutput) {
        throw new Error("No output received from stream().");
      }

      setStatus("ready");
      const lastRound = finalOutput.runs[finalOutput.runs.length - 1];
      addLog(
        `Agent stream complete. done=${String(finalOutput.done)}, rounds=${finalOutput.runs.length}, totalTokens=${finalOutput.usage?.totalTokens ?? 0}, lastRunTokens=${lastRound?.usage?.totalTokens ?? 0}`,
      );
    } catch (error) {
      const message = errorMessage(error);
      setResponses((prev) =>
        prev.map((response) =>
          response.id === responseId
            ? { ...response, done: true, error: message }
            : response,
        ),
      );
      setStatus("error");
      addLog(message);
    }
  };

  return (
    <main className="min-h-screen bg-canvas px-4 py-8 text-ink sm:px-8">
      <div className="mx-auto w-full max-w-3xl rounded-2xl border border-ink/15 bg-white/70 p-6 shadow-soft backdrop-blur-sm sm:p-8">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Transformers Agent SDK Playground
        </h1>
        <p className="mt-2 text-sm text-ink/75">
          A tiny React + Vite + Tailwind app to smoke test `Model` and `Agent`.
        </p>

        <section className="mt-6 space-y-4">
          <label className="block">
            <span className="mb-1 block text-sm font-medium">Model ID</span>
            <input
              className="w-full rounded-lg border border-ink/25 bg-white px-3 py-2 outline-none ring-accent/30 focus:ring"
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-sm font-medium">Prompt</span>
            <textarea
              className="h-24 w-full rounded-lg border border-ink/25 bg-white px-3 py-2 outline-none ring-accent/30 focus:ring"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </label>

          <div className="flex flex-wrap gap-2">
            <button
              className="rounded-lg bg-ink px-3 py-2 text-sm text-white"
              onClick={checkCache}
            >
              Check Cache
            </button>
            <button
              className="rounded-lg bg-pine px-3 py-2 text-sm text-white"
              onClick={initModel}
            >
              Initialize Model
            </button>
            <button
              className="rounded-lg bg-accent px-3 py-2 text-sm text-white"
              onClick={runAgent}
            >
              Run Agent
            </button>
          </div>
        </section>

        <section className="mt-6 grid gap-3 rounded-xl border border-ink/15 bg-white p-4 text-sm sm:grid-cols-3">
          <Info label="Status" value={status} />
          <Info
            label="Cached"
            value={cached === null ? "unknown" : String(cached)}
          />
          <Info
            label="Missing"
            value={downloadBytes === null ? "-" : formatBytes(downloadBytes)}
          />
          <Info
            label="In Cache"
            value={cachedBytes === null ? "-" : formatBytes(cachedBytes)}
          />
        </section>

        <section className="mt-6">
          <h2 className="text-sm font-semibold">Conversation</h2>
          {responses.length === 0 ? (
            <pre className="mt-2 min-h-20 rounded-lg border border-ink/15 bg-white p-3 text-sm whitespace-pre-wrap">
              No messages yet.
            </pre>
          ) : (
            <div className="mt-3 space-y-6 rounded-2xl border border-ink/10 bg-ink/[0.03] p-4">
              {responses.map((response) => (
                <div key={response.id} className="space-y-3">
                  <div className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-3 text-sm text-white shadow-sm">
                      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-white/70">
                        You
                      </div>
                      <div className="whitespace-pre-wrap">
                        {response.prompt}
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-start">
                    <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-ink/10 bg-white px-4 py-3 text-sm shadow-sm">
                      <div className="mb-2 flex items-center justify-between gap-3 text-xs uppercase tracking-wide text-ink/50">
                        <span>Assistant</span>
                        <span>{response.done ? "done" : "streaming"}</span>
                      </div>
                      {response.error ? (
                        <pre className="rounded-md bg-red-50 p-2 text-xs text-red-700 whitespace-pre-wrap">
                          {response.error}
                        </pre>
                      ) : null}
                      <div className="min-h-6 whitespace-pre-wrap">
                        {response.text || "Thinking..."}
                      </div>
                      {response.result === null ? null : (
                        <div className="mt-4 space-y-2 border-t border-ink/10 pt-3">
                          {response.result.runs.map((round, index) => (
                            <div
                              key={`${response.id}-${index}`}
                              className="rounded-lg bg-ink/[0.03] p-3"
                            >
                              <div className="text-xs font-medium uppercase tracking-wide text-ink/50">
                                Round {index + 1}
                              </div>
                              {round.thinkingText ? (
                                <pre className="mt-2 rounded-md bg-white/80 p-2 text-xs whitespace-pre-wrap text-ink/75">
                                  {round.thinkingText}
                                </pre>
                              ) : null}
                              {round.text ? (
                                <pre className="mt-2 rounded-md bg-white/80 p-2 text-xs whitespace-pre-wrap">
                                  {round.text}
                                </pre>
                              ) : null}
                              {round.tools.length > 0 ? (
                                <pre className="mt-2 rounded-md bg-white/80 p-2 text-xs whitespace-pre-wrap">
                                  {JSON.stringify(round.tools, null, 2)}
                                </pre>
                              ) : null}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="mt-6">
          <h2 className="text-sm font-semibold">Recent Log</h2>
          <ul className="mt-2 space-y-1 text-xs text-ink/80">
            {log.length === 0 ? (
              <li>No events yet.</li>
            ) : (
              log.map((line, index) => <li key={`${index}-${line}`}>{line}</li>)
            )}
          </ul>
        </section>
      </div>
    </main>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-ink/60">{label}</div>
      <div className="mt-1 font-medium">{value}</div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(value >= 100 || idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
