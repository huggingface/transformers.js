import { useEffect, useRef, useState } from "react";

type Status =
  | "loading"
  | "initiate"
  | "progress"
  | "done"
  | "ready"
  | "start"
  | "update"
  | "complete"
  | "error";

interface ProgressItem {
  file: string;
  loaded: number;
  progress: number;
  total: number;
  name?: string;
  status: string;
}

export const useWorker = () => {
  const worker = useRef<Worker | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Model loading and progress
  const [status, setStatus] = useState<Status | null>(null);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);

  // Inputs and outputs
  const [text, setText] = useState("");
  const [tps, setTps] = useState<number | null>(null);
  const [language, setLanguage] = useState("en");

  // Processing
  const [recording, setRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chunks, setChunks] = useState<Blob[]>([]);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL("../worker.js", import.meta.url), {
        type: "module",
      });
    } else {
      return;
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e: MessageEvent) => {
      switch (e.data.status) {
        case "loading":
          // Model file start load: add a new progress item to the list.
          setStatus("loading");
          setLoadingMessage(e.data.data);
          break;

        case "initiate":
          setProgressItems((prev) => [...prev, e.data]);
          break;

        case "progress":
          // Model file progress: update one of the progress items.
          setProgressItems((prev) =>
            prev.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data };
              }
              return item;
            })
          );
          break;

        case "done":
          // Model file loaded: remove the progress item from the list.
          setProgressItems((prev) =>
            prev.filter((item) => item.file !== e.data.file)
          );
          break;

        case "ready":
          // Pipeline ready: the worker is ready to accept messages.
          setStatus("ready");
          if (
            recorderRef.current &&
            recorderRef.current.state !== "recording"
          ) {
            recorderRef.current.start();
          }
          break;

        case "start":
          // Start generation
          setIsProcessing(true);

          // Request new data from the recorder
          if (recorderRef.current) {
            recorderRef.current.requestData();
          }
          break;

        case "update":
          {
            const { tps, output } = e.data;
            setText(output || "");
            setTps(tps);
          }
          // Generation update: update the output text.
          break;

        case "complete":
          // Generation complete: re-enable the "Generate" button
          setIsProcessing(false);
          setText(e.data.output);
          break;
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current?.removeEventListener("message", onMessageReceived);
      worker.current = null;
    };
  }, []);

  return {
    worker: worker.current,
    status,
    loadingMessage,
    progressItems,
    audioContextRef,
    chunks,
    tps,
    recorderRef,
    isProcessing,
    recording,
    text,
    language,
    stream,
    setStatus,
    setStream,
    setRecording,
    setChunks,
    setLanguage,
  };
};
