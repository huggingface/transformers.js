import { useEffect, useState, useRef, useCallback } from 'react';

import Progress from './components/Progress';
import ImageInput from './components/ImageInput';

const IS_WEBGPU_AVAILABLE = !!navigator.gpu;

function App() {

  // Create a reference to the worker object.
  const worker = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [progressItems, setProgressItems] = useState([]);

  const [task, setTask] = useState('<CAPTION>');
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [time, setTime] = useState(null);

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
    if (status === null) {
      setStatus('loading');
      worker.current.postMessage({ type: 'load' });
    } else {
      setStatus('running');
      worker.current.postMessage({
        type: 'run', data: { text, url: image, task }
      });
    }
  }, [status, task, image, text]);

  return (
    IS_WEBGPU_AVAILABLE
      ? (<div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900 max-w-[630px]">

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
        <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
          <div className="flex flex-col items-center mb-1 text-center">
            <h1 className="text-6xl font-bold mb-2">Florence2 WebGPU</h1>
            <h2 className="text-xl font-semibold">Powerful vision foundation model running locally in your browser.</h2>
          </div>

          <div className="w-full min-h-[220px] flex flex-col justify-center items-center p-2">

            <p className="mb-2">
              You are about to download <a href="https://huggingface.co/onnx-community/Florence-2-base-ft" target="_blank" rel="noreferrer" className="font-medium underline">Florence-2-base-ft</a>,
              a 230 million parameter vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks like captioning, object detection, and segmentation.
              Once loaded, the model (340&nbsp;MB) will be cached and reused when you revisit the page.<br />
              <br />
              Everything runs locally in your browser using <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" className="underline">🤗&nbsp;Transformers.js</a> and ONNX Runtime Web,
              meaning no API calls are made to a server for inference. You can even disconnect from the internet after the model has loaded!
            </p>

            <div className="flex w-full justify-around m-4">
              <div className="flex flex-col gap-2 w-full max-w-[48%]">
                <div className="flex flex-col">
                  <span className="text-sm mb-0.5">Task</span>
                  <select
                    className="border rounded-md p-1"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                  >
                    <option value="<CAPTION>">Caption</option>
                    <option value="<DETAILED_CAPTION>">Detailed Caption</option>
                    <option value="<MORE_DETAILED_CAPTION>">More Detailed Caption</option>
                    <option value="<OCR>">OCR</option>
                    <option value="<OCR_WITH_REGION>">OCR with Region</option>
                    <option value="<OD>">Object Detection</option>
                    <option value="<DENSE_REGION_CAPTION>">Dense Region Caption</option>
                    <option value="<CAPTION_TO_PHRASE_GROUNDING>">Caption to Phrase Grounding</option>
                    {/* <option value="<REFERRING_EXPRESSION_SEGMENTATION>">Referring Expression Segmentation</option> */}
                    {/* <option value="<REGION_TO_SEGMENTATION>">Region to Segmentation</option> */}
                    {/* <option value="<OPEN_VOCABULARY_DETECTION>">Open Vocabulary Detection</option> */}
                    {/* <option value="<REGION_TO_CATEGORY>">Region to Category</option> */}
                    {/* <option value="<REGION_TO_DESCRIPTION>">Region to Description</option> */}
                    {/* <option value="<REGION_TO_OCR>">Region to OCR</option> */}
                    {/* <option value="<REGION_PROPOSAL>">Region Proposal</option> */}
                  </select>
                </div>
                <div className="flex flex-col">
                  <span className="text-sm mb-0.5">Input Image</span>
                  <ImageInput className="flex flex-col items-center border border-gray-300 rounded-md cursor-pointer h-[250px]" onImageChange={(file, result) => {
                    worker.current.postMessage({ type: 'reset' }); // Reset image cache
                    setResult(null);
                    setImage(result);
                  }} />
                </div>
              </div>
              <div className="flex flex-col gap-2 w-full max-w-[48%] justify-end">
                {
                  task === '<CAPTION_TO_PHRASE_GROUNDING>'
                  && (<div className="flex flex-col">
                    <span className="text-sm mb-0.5">Text input</span>
                    <input className="border rounded-md px-2 py-[3.5px]"
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                    />
                  </div>)
                }

                <div className="flex flex-col relative">
                  <span className="text-sm mb-0.5">Output</span>
                  <div className="flex justify-center border border-gray-300 rounded-md h-[250px]">
                    {result?.[task] && (<>
                      {
                        typeof result[task] === 'string'
                          ? <p className="pt-4 px-4 text-center max-h-[205px] overflow-y-auto">{result[task]}</p>
                          : <pre className="w-full h-full p-2 overflow-y-auto">
                            {JSON.stringify(result[task], null, 2)}
                          </pre>
                      }
                      {
                        time && <p className="text-sm text-gray-500 absolute bottom-2 bg-white p-1 rounded border">Execution time: {time.toFixed(2)} ms</p>
                      }
                    </>)
                    }
                  </div>

                </div>
              </div>
            </div>

            <button
              className="border px-4 py-2 rounded-lg bg-blue-400 text-white hover:bg-blue-500 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
              onClick={handleClick}
              disabled={status === 'running' || (status !== null && image === null)}
            >
              {status === null ? 'Load model' :
                status === 'running'
                  ? 'Running...'
                  : 'Run model'
              }
            </button>
          </div>
        </div>

      </div >)
      : (<div className="fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center">WebGPU is not supported<br />by this browser :&#40;</div>)
  )
}

export default App
