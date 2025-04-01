import { useEffect, useRef, useState } from "react";
import AudioVisualizer from "./AudioVisualizer";
import { useWorker } from "../hooks/useWorker";
import Progress from "./Progress";

import { LanguageSelector } from "./LanguageSelector";

const IS_WEBGPU_AVAILABLE = !!navigator.gpu;

const WHISPER_SAMPLING_RATE = 16_000;
const MAX_AUDIO_LENGTH = 30; // seconds
const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH;

export default function Whisper() {
  const {
    worker,
    status,
    loadingMessage,
    progressItems,
    audioContextRef,
    chunks,
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
    tps,
  } = useWorker();

  // State for incremental word display
  const [displayedText, setDisplayedText] = useState<string>("");
  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (text) {
      setDisplayedText((prev) => prev + text);
      if (textAreaRef.current) {
        textAreaRef.current.scrollTop = textAreaRef.current.scrollHeight;
      }
    }
  }, [text]);

  useEffect(() => {
    if (recorderRef.current) return;

    const initializeMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        setStream(stream);

        const recorder = new MediaRecorder(stream);
        const audioContext = new AudioContext({
          sampleRate: WHISPER_SAMPLING_RATE,
        });
        recorderRef.current = recorder;
        audioContextRef.current = audioContext;

        recorder.onstart = () => {
          setRecording(true);
          setChunks([]);
          // Reset displayed text when starting a new recording
          setDisplayedText("");
        };

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            setChunks((prev) => [...prev, e.data]);
          } else {
            setTimeout(() => recorder.requestData(), 25);
          }
        };

        recorder.onstop = () => setRecording(false);
      } catch (error) {
        console.error("Media initialization failed:", error);
      }
    };

    initializeMedia();

    return () => {
      if (recorderRef.current) {
        if (recorderRef.current.state === "recording") {
          recorderRef.current.stop();
        }
        recorderRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!recorderRef.current) return;
    if (!recording) return;
    if (isProcessing) return;
    if (status !== "ready") return;

    if (chunks.length > 0) {
      // Generate from data
      const blob = new Blob(chunks, { type: recorderRef.current.mimeType });

      const fileReader = new FileReader();

      fileReader.onloadend = async () => {
        const arrayBuffer = fileReader.result;
        try {
          const decoded = await audioContextRef.current?.decodeAudioData(
            arrayBuffer as ArrayBuffer
          );
          if (decoded) {
            let audio = decoded.getChannelData(0);
            if (audio.length > MAX_SAMPLES) {
              // Get last MAX_SAMPLES
              audio = audio.slice(-MAX_SAMPLES);
            }

            worker?.postMessage({
              type: "generate",
              data: { audio, language },
            });
          }
        } catch (error) {
          console.error("Audio decoding failed:", error);
        }
      };
      fileReader.readAsArrayBuffer(blob);
    } else {
      recorderRef.current?.requestData();
    }
  }, [
    status,
    recording,
    isProcessing,
    chunks,
    language,
    worker,
    audioContextRef,
    recorderRef,
  ]);

  const handleStartTranscribing = () => {
    setStatus("loading");
    worker?.postMessage({ type: "load" });
  };

  return IS_WEBGPU_AVAILABLE ? (
    <div className="flex flex-col h-screen mx-auto justify-end bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 text-gray-800 dark:text-gray-200">
      <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
        <div className="flex flex-col items-center mb-6 max-w-[500px] text-center">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-full shadow-lg mb-4">
            <img
              src="logo.png"
              width="100px"
              height="auto"
              className="block"
              alt="Logo"
            />
          </div>
          <h1 className="text-5xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">
            Whisper WebGPU
          </h1>
          <h2 className="text-xl font-semibold">
            Real-time in-browser speech recognition
          </h2>
        </div>

        <div className="flex flex-col items-center px-4 w-full max-w-[600px]">
          {status === null && (
            <>
              <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md mb-6 max-w-[520px]">
                <p className="leading-relaxed">
                  You are about to load{" "}
                  <a
                    href="https://huggingface.co/onnx-community/whisper-base"
                    target="_blank"
                    rel="noreferrer"
                    className="font-medium text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    whisper-base
                  </a>
                  , a 73 million parameter speech recognition model that is
                  optimized for inference on the web. Once downloaded, the model
                  (~200&nbsp;MB) will be cached and reused when you revisit the
                  page.
                  <br />
                  <br />
                  Everything runs directly in your browser using{" "}
                  <a
                    href="https://huggingface.co/docs/transformers.js"
                    target="_blank"
                    rel="noreferrer"
                    className="font-medium text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    🤗&nbsp;Transformers.js
                  </a>{" "}
                  and ONNX Runtime Web, meaning no data is sent to a server. You
                  can even disconnect from the internet after the model has
                  loaded!
                </p>
              </div>
              <div className="flex gap-5">
                <button
                  className="px-6 py-3 rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:bg-blue-300 disabled:cursor-not-allowed select-none transition-colors shadow-md"
                  onClick={handleStartTranscribing}
                  disabled={status !== null}
                >
                  Load model
                </button>
                <LanguageSelector
                  language={language}
                  setLanguage={setLanguage}
                />
              </div>
            </>
          )}

          <div className="w-full max-w-[600px] p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-4 mb-4">
              <AudioVisualizer
                className="w-full h-[120px] rounded-lg"
                stream={stream!}
              />
            </div>

            {status === "ready" && (
              <div className="relative bg-white dark:bg-gray-800 rounded-xl shadow-md p-4">
                <textarea
                  aria-label="text"
                  value={displayedText || "Speak to see transcription..."}
                  ref={textAreaRef}
                  className="w-full min-h-[120px] max-h-[200px] overflow-y-auto overflow-wrap-anywhere rounded-lg p-2 bg-gray-50 dark:bg-gray-700"
                >
                  {!displayedText && recording ? (
                    <span>
                      Listening<span className="animate-pulse">...</span>
                    </span>
                  ) : (
                    <span>
                      Listening<span className="animate-pulse">...</span>
                    </span>
                  )}
                </textarea>
                {tps && (
                  <div className="absolute bottom-2 right-4 px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full text-xs font-medium">
                    {tps?.toFixed(2)} tok/s
                  </div>
                )}
              </div>
            )}

            {status === "loading" && (
              <div className="w-full max-w-[600px] bg-white dark:bg-gray-800 rounded-xl shadow-md p-6 text-left mx-auto mt-4">
                <p className="text-center font-medium mb-4">{loadingMessage}</p>
                {progressItems.map(({ file, progress, total }, i) => (
                  <Progress
                    key={i}
                    text={file}
                    percentage={progress}
                    total={total}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  ) : (
    <div className="fixed w-screen h-screen bg-gradient-to-br from-gray-900 to-black z-10 text-white text-2xl font-semibold flex flex-col justify-center items-center text-center p-6">
      <div className="bg-red-500/20 p-6 rounded-xl border border-red-500/30 max-w-md">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-16 w-16 mx-auto mb-4 text-red-500"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <h2 className="text-3xl font-bold mb-2">WebGPU Not Supported</h2>
        <p className="text-lg">
          This browser doesn't support WebGPU, which is required for this
          application. Please try using a modern browser like Chrome or Edge.
        </p>
      </div>
    </div>
  );
}
