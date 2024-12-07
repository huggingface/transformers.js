import { useEffect, useState, useRef, useCallback } from 'react';

import Progress from './components/Progress';
import MediaInput from './components/MediaInput';
import Transcript from './components/Transcript';
import LanguageSelector from './components/LanguageSelector';


async function hasWebGPU() {
  if (!navigator.gpu) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}

function App() {

  // Create a reference to the worker object.
  const worker = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [progressItems, setProgressItems] = useState([]);

  const mediaInputRef = useRef(null);
  const [audio, setAudio] = useState(null);
  const [language, setLanguage] = useState('en');

  const [result, setResult] = useState(null);
  const [time, setTime] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);

  const [device, setDevice] = useState('webgpu'); // Try use WebGPU first
  const [modelSize, setModelSize] = useState('gpu' in navigator ? 196 : 77); // WebGPU=196MB, WebAssembly=77MB
  useEffect(() => {
    hasWebGPU().then((result) => {
      setModelSize(result ? 196 : 77);
      setDevice(result ? 'webgpu' : 'wasm');
    });
  }, []);

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
          break;

        case 'complete':
          setResult(e.data.result);
          setTime(e.data.time);
          setStatus('ready');
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

  const handleClick = useCallback(() => {
    setResult(null);
    setTime(null);
    if (status === null) {
      setStatus('loading');
      worker.current.postMessage({ type: 'load', data: { device } });
    } else {
      setStatus('running');
      worker.current.postMessage({
        type: 'run', data: { audio, language }
      });
    }
  }, [status, audio, language, device]);

  return (
    <div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900 max-w-[560px]">

      {status === 'loading' && (
        <div className="flex justify-center items-center fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] top-0 left-0">
          <div className="w-[500px]">
            <p className="text-center mb-1 text-white text-md">{loadingMessage}</p>
            {progressItems.map(({ file, progress, total }, i) => (
              <Progress key={i} text={file} percentage={progress} total={total} />
            ))}
          </div>
        </div>
      )}
      <div className="h-full flex justify-center items-center flex-col relative">
        <div className="flex flex-col items-center mb-1 text-center">
          <h1 className="text-5xl font-bold mb-2">Whisper Timestamped</h1>
          <h2 className="text-xl font-semibold">In-browser speech recognition w/ word-level timestamps</h2>
        </div>

        <div className="w-full min-h-[220px] flex flex-col justify-center items-center p-2">
          {
            !audio && (
              <p className="mb-2">
                You are about to download <a href="https://huggingface.co/onnx-community/whisper-base_timestamped" target="_blank" rel="noreferrer" className="font-medium underline">whisper-base (timestamped)</a>,
                a 73 million parameter speech recognition model with the ability to generate word-level timestamps across 100 different languages.
                Once loaded, the model ({modelSize}&nbsp;MB) will be cached and reused when you revisit the page.<br />
                <br />
                Everything runs locally in your browser using <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" className="underline">🤗&nbsp;Transformers.js</a> and ONNX Runtime Web,
                meaning no API calls are made to a server for inference. You can even disconnect from the internet after the model has loaded!
              </p>
            )
          }

          <div className="flex flex-col w-full m-3">
            <span className="text-sm mb-0.5">Input audio/video</span>
            <MediaInput
              ref={mediaInputRef}
              className="flex items-center border rounded-md cursor-pointer min-h-[100px] max-h-[500px] overflow-hidden"
              onInputChange={(result) => setAudio(result)}
              onTimeUpdate={(time) => setCurrentTime(time)}
            />
          </div>

          <div className="relative w-full flex justify-center items-center">
            <button
              className="border px-4 py-2 rounded-lg bg-blue-400 text-white hover:bg-blue-500 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
              onClick={handleClick}
              disabled={status === 'running' || (status !== null && audio === null)}
            >
              {status === null ? 'Load model' :
                status === 'running'
                  ? 'Running...'
                  : 'Run model'
              }
            </button>

            {status !== null &&
              <div className='absolute right-0 bottom-0'>
                <span className="text-xs">Language:</span>
                <br />
                <LanguageSelector className="border rounded-lg p-1 max-w-[100px]" language={language} setLanguage={setLanguage} />
              </div>
            }
          </div>

          {
            result && time && (
              <>
                <div className="w-full mt-4 border rounded-md">
                  <Transcript
                    className="p-2 max-h-[200px] overflow-y-auto scrollbar-thin select-none"
                    transcript={result}
                    currentTime={currentTime}
                    setCurrentTime={(time) => {
                      setCurrentTime(time);
                      mediaInputRef.current.setMediaTime(time);
                    }}
                  />
                </div>
                <p className="text-sm text-gray-600 text-end p-1">Generation time: <span className="text-gray-800 font-semibold">{time.toFixed(2)}ms</span></p>
              </>
            )


          }
        </div>
      </div>

    </div >
  )
}

export default App
