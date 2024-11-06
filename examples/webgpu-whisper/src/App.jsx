import { useEffect, useState, useRef } from 'react';

import { AudioVisualizer } from './components/AudioVisualizer';
import Progress from './components/Progress';
import { LanguageSelector } from './components/LanguageSelector';

const IS_WEBGPU_AVAILABLE = !!navigator.gpu;

const WHISPER_SAMPLING_RATE = 16_000;
const MAX_AUDIO_LENGTH = 30; // seconds
const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH;

function App() {

  // Create a reference to the worker object.
  const worker = useRef(null);

  const recorderRef = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [progressItems, setProgressItems] = useState([]);

  // Inputs and outputs
  const [text, setText] = useState('');
  const [tps, setTps] = useState(null);
  const [language, setLanguage] = useState('en');

  // Processing
  const [recording, setRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [stream, setStream] = useState(null);
  const audioContextRef = useRef(null);

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module'
      });
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case 'loading':
          // Model file start load: add a new progress item to the list.
          setStatus('loading');
          setLoadingMessage(e.data.data);
          break;

        case 'initiate':
          setProgressItems(prev => [...prev, e.data]);
          break;

        case 'progress':
          // Model file progress: update one of the progress items.
          setProgressItems(
            prev => prev.map(item => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data }
              }
              return item;
            })
          );
          break;

        case 'done':
          // Model file loaded: remove the progress item from the list.
          setProgressItems(
            prev => prev.filter(item => item.file !== e.data.file)
          );
          break;

        case 'ready':
          // Pipeline ready: the worker is ready to accept messages.
          setStatus('ready');
          recorderRef.current?.start();
          break;

        case 'start': {
          // Start generation
          setIsProcessing(true);

          // Request new data from the recorder
          recorderRef.current?.requestData();
        }
          break;

        case 'update': {
          // Generation update: update the output text.
          const { tps } = e.data;
          setTps(tps);
        }
          break;

        case 'complete':
          // Generation complete: re-enable the "Generate" button
          setIsProcessing(false);
          setText(e.data.output);
          break;
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener('message', onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener('message', onMessageReceived);
    };
  }, []);

  useEffect(() => {
    if (recorderRef.current) return; // Already set

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          setStream(stream);

          recorderRef.current = new MediaRecorder(stream);
          audioContextRef.current = new AudioContext({ sampleRate: WHISPER_SAMPLING_RATE });

          recorderRef.current.onstart = () => {
            setRecording(true);
            setChunks([]);
          }
          recorderRef.current.ondataavailable = (e) => {
            if (e.data.size > 0) {
              setChunks((prev) => [...prev, e.data]);
            } else {
              // Empty chunk received, so we request new data after a short timeout
              setTimeout(() => {
                recorderRef.current.requestData();
              }, 25);
            }
          };

          recorderRef.current.onstop = () => {
            setRecording(false);
          };

        })
        .catch(err => console.error("The following error occurred: ", err));
    } else {
      console.error("getUserMedia not supported on your browser!");
    }

    return () => {
      recorderRef.current?.stop();
      recorderRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!recorderRef.current) return;
    if (!recording) return;
    if (isProcessing) return;
    if (status !== 'ready') return;

    if (chunks.length > 0) {
      // Generate from data
      const blob = new Blob(chunks, { type: recorderRef.current.mimeType });

      const fileReader = new FileReader();

      fileReader.onloadend = async () => {
        const arrayBuffer = fileReader.result;
        const decoded = await audioContextRef.current.decodeAudioData(arrayBuffer);
        let audio = decoded.getChannelData(0);
        if (audio.length > MAX_SAMPLES) { // Get last MAX_SAMPLES
          audio = audio.slice(-MAX_SAMPLES);
        }

        worker.current.postMessage({ type: 'generate', data: { audio, language } });
      }
      fileReader.readAsArrayBuffer(blob);
    } else {
      recorderRef.current?.requestData();
    }
  }, [status, recording, isProcessing, chunks, language]);

  return (
    IS_WEBGPU_AVAILABLE
      ? (<div className="flex flex-col h-screen mx-auto justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900">
        {(
          <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
            <div className="flex flex-col items-center mb-1 max-w-[400px] text-center">
              <img src="logo.png" width="50%" height="auto" className="block"></img>
              <h1 className="text-4xl font-bold mb-1">Whisper WebGPU</h1>
              <h2 className="text-xl font-semibold">Real-time in-browser speech recognition</h2>
            </div>

            <div className="flex flex-col items-center px-4">
              {status === null && (<>
                <p className="max-w-[480px] mb-4">
                  <br />
                  You are about to load <a href="https://huggingface.co/onnx-community/whisper-base" target="_blank" rel="noreferrer" className="font-medium underline">whisper-base</a>,
                  a 73 million parameter speech recognition model that is optimized for inference on the web. Once downloaded, the model (~200&nbsp;MB) will be cached and reused when you revisit the page.<br />
                  <br />
                  Everything runs directly in your browser using <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" className="underline">🤗&nbsp;Transformers.js</a> and ONNX Runtime Web,
                  meaning no data is sent to a server. You can even disconnect from the internet after the model has loaded!
                </p>

                <button
                  className="border px-4 py-2 rounded-lg bg-blue-400 text-white hover:bg-blue-500 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
                  onClick={() => {
                    worker.current.postMessage({ type: 'load' });
                    setStatus('loading');
                  }}
                  disabled={status !== null}
                >
                  Load model
                </button>
              </>)}

              <div className="w-[500px] p-2">
                <AudioVisualizer className="w-full rounded-lg" stream={stream} />
                {status === 'ready' && <div className="relative">
                  <p className="w-full h-[80px] overflow-y-auto overflow-wrap-anywhere border rounded-lg p-2">{text}</p>
                  {tps && <span className="absolute bottom-0 right-0 px-1">{tps.toFixed(2)} tok/s</span>}
                </div>}

              </div>
              {status === 'ready' && <div className='relative w-full flex justify-center'>
                <LanguageSelector language={language} setLanguage={(e) => {
                  recorderRef.current?.stop();
                  setLanguage(e);
                  recorderRef.current?.start();
                }} />
                <button className="border rounded-lg px-2 absolute right-2" onClick={() => {
                  recorderRef.current?.stop();
                  recorderRef.current?.start();
                }}>Reset</button>
              </div>
              }
              {status === 'loading' && (
                <div className="w-full max-w-[500px] text-left mx-auto p-4">
                  <p className="text-center">{loadingMessage}</p>
                  {progressItems.map(({ file, progress, total }, i) => (
                    <Progress key={i} text={file} percentage={progress} total={total} />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>)
      : (<div className="fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center">WebGPU is not supported<br />by this browser :&#40;</div>)
  )
}

export default App
