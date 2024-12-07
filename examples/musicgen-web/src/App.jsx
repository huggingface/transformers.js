import { useEffect, useState, useRef } from 'react';
import { AutoTokenizer, MusicgenForConditionalGeneration, BaseStreamer } from '@xenova/transformers';
import { encodeWAV, share } from './utils.js';

import './App.css';

const MODEL_ID = 'Xenova/musicgen-small';

// Adapted from https://huggingface.co/spaces/facebook/MusicGen
const EXAMPLES = [
  '80s pop track with bassy drums and synth',
  '90s rock song with loud guitars and heavy drums',
  'a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130',
  'A cheerful country song with acoustic guitars',
  'lofi slow bpm electro chill with organic samples',
];

// Enable sharing if running on Hugging Face Spaces
const SHARING_ENABLED = window.location.host.endsWith('.hf.space');

// Streamer to update progress
class CallbackStreamer extends BaseStreamer {
  constructor(callback_fn) {
    super();
    this.callback_fn = callback_fn;
  }

  put(value) {
    return this.callback_fn(value);
  }

  end() {
    return this.callback_fn();
  }
}

// Main App component
const App = () => {
  // Input/output state
  const [textInput, setTextInput] = useState(EXAMPLES[0]);
  const [progress, setProgress] = useState(0);
  const [loadProgress, setLoadProgress] = useState({});
  const [statusText, setStatusText] = useState('Loading model (656MB)...');
  const [result, setResult] = useState(null);
  const audioRef = useRef(null);

  // Model and tokenizer references
  const modelPromise = useRef(null);
  const tokenizerPromise = useRef(null);

  // Generation parameters
  const [guidance_scale, setGuidanceScale] = useState(3);
  const [temperature, setTemperature] = useState(1);
  const [duration, setDuration] = useState(10);

  // Load model and tokenizer on first render
  useEffect(() => {
    modelPromise.current ??= MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, {
      progress_callback: (data) => {
        if (data.status !== 'progress') return;
        setLoadProgress(prev => ({ ...prev, [data.file]: data }))
      },
      dtype: {
        text_encoder: 'q8',
        decoder_model_merged: 'q8',
        encodec_decode: 'fp32',
      },
      device: 'wasm',
    });

    tokenizerPromise.current ??= AutoTokenizer.from_pretrained(MODEL_ID);
  }, []);

  // Update progress bar based on load progress
  useEffect(() => {
    const items = Object.values(loadProgress);
    if (items.length !== 5) return; // 5 files to load
    let loaded = 0;
    let total = 0;
    for (const data of Object.values(loadProgress)) {
      loaded += data.loaded;
      total += data.total;
    }
    const progress = loaded / total;
    setProgress(progress);
    setStatusText(progress === 1
      ? 'Ready!'
      : `Loading model (${(progress * 100).toFixed()}% of 656MB)...`
    );
  }, [loadProgress]);

  // Function to handle generating music
  const generateMusic = async () => {
    // Reset audio player and result
    audioRef.current.src = '';
    setResult(null);

    // Get model and tokenizer
    const tokenizer = await tokenizerPromise.current;
    const model = await modelPromise.current;

    // Get number of tokens to match user-specified duration (more intuitive for user)
    // 503 tokens -> 10 seconds generated => ~50 tokens per second
    // https://huggingface.co/docs/transformers/model_doc/musicgen#generation
    const max_length = Math.min(
      Math.max(Math.floor(duration * 50), 1) + 4,
      model.generation_config.max_length ?? 1500,
    );

    // Create a streamer to update progress
    let num_tokens = 0;
    const streamer = new CallbackStreamer((value) => {
      const percent = value === undefined ? 1 : ++num_tokens / max_length;
      setStatusText(`Generating (${(percent * 100).toFixed()}%)...`);
      setProgress(percent);
    });

    // Tokenize input text
    const inputs = tokenizer(textInput);

    // Generate music
    const audio_values = await model.generate({
      // Inputs
      ...inputs,

      // Generation parameters
      max_length,
      guidance_scale,
      temperature,

      // Outputs
      streamer,
    });

    setStatusText('Encoding audio...');

    // Encode audio values to WAV
    const sampling_rate = model.config.audio_encoder.sampling_rate;
    const wav = encodeWAV(audio_values.data, sampling_rate);
    const blob = new Blob([wav], { type: 'audio/wav' });
    setResult(blob);

    audioRef.current.src = URL.createObjectURL(blob);
    setStatusText('Done!');
  };

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-5xl font-bold mb-2">MusicGen Web</h1>
      <h2 className="text-2xl font-semibold mb-4">In-browser text-to-music w/ <a className="underline" href="https://github.com/huggingface/transformers.js">🤗 Transformers.js!</a>
      </h2>

      {/* Text input for user */}
      <input
        type="text"
        placeholder="Describe the music to generate..."
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
        className="border border-gray-300 p-2 mb-4 w-full rounded"
      />

      {/* Example buttons */}
      <div className="mb-4 flex gap-2 justify-center text-sm">
        {EXAMPLES.map((example, i) => (
          <button key={i} className="bg-blue-500 hover:bg-blue-400 transition-colors duration-100 text-white px-2 py-2 rounded" onClick={(e) => setTextInput(e.target.innerText)}>{example}</button>
        ))}
      </div>

      {/* Generation parameters */}
      <div className="flex mb-4 justify-center gap-2">
        {/* Duration */}
        <div>
          <label className="block text-sm font-semibold mb-1">Duration</label>
          <input type="range" min={1} max={30} value={duration} onChange={(e) => setDuration(e.target.value)} />
          <p className="text-sm text-center">{`${duration} second${duration > 1 ? 's' : ''}`}</p>
        </div>

        {/* Guidance Scale */}
        <div className="mr-4">
          <label className="block text-sm font-semibold mb-1">Guidance Scale</label>
          <input type="range" min={1} max={10} value={guidance_scale} onChange={(e) => setGuidanceScale(e.target.value)} />
          <p className="text-sm text-center">{guidance_scale}</p>
        </div>

        {/* Temperature */}
        <div>
          <label className="block text-sm font-semibold mb-1">Temperature</label>
          <input type="range" min={0.1} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(e.target.value)} />
          <p className="text-sm text-center">{temperature}</p>
        </div>
      </div>

      {/* Button to generate music */}
      <button className="mb-4 bg-green-500 hover:bg-green-400 transition-colors duration-100 text-white px-4 py-3 rounded-lg font-semibold" onClick={generateMusic}>Generate Music</button>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="bg-gray-200 h-4 w-full rounded-full">
          <div className="bg-blue-500 h-4 rounded-full" style={{ width: `${100 * progress}%` }}></div>
        </div>
        <p className="text-sm text-center mt-1">{statusText}</p>
      </div>

      {/* Audio player */}
      {<div className="flex justify-center flex-col items-center">
        <audio ref={audioRef} controls type="audio/wav" />
        {SHARING_ENABLED && result &&
          <button
            className="bg-red-500 hover:bg-red-400 transition-colors duration-100 text-white px-2 py-1 my-2 rounded-lg text-sm"
            onClick={async (e) => {
              e.target.disabled = true;
              e.target.innerText = 'Uploading...';
              await share(result, {
                prompt: textInput,
                duration,
                guidance_scale,
                temperature,
              });
              e.target.disabled = false;
              e.target.innerText = 'Share';
            }
            }>Share</button>
        }
      </div>}
    </div>
  );
};

export default App;
