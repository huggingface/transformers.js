# @huggingface/transformers-react

React hooks for [Transformers.js](https://github.com/huggingface/transformers.js) - Run 🤗 Transformers directly in your React applications!

## Installation

```bash
npm install @huggingface/transformers @huggingface/transformers-react
```

## Usage

### Basic Example

```tsx
import { usePipeline } from '@huggingface/transformers-react';
import { useState } from 'react';

function SentimentAnalysis() {
  const { pipeline, loading, error, run } = usePipeline({
    task: 'sentiment-analysis',
  });

  const [result, setResult] = useState(null);
  const [input, setInput] = useState('');

  const handleAnalyze = async () => {
    const output = await run(input);
    setResult(output);
  };

  if (loading) return <div>Loading model...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h1>Sentiment Analysis</h1>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter text to analyze"
      />
      <button onClick={handleAnalyze}>Analyze</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
```

### Using Custom Models

```tsx
const { pipeline, loading, error, run } = usePipeline({
  task: 'text-generation',
  model: 'Xenova/gpt2',
});
```

### Available Pipelines

All Transformers.js pipeline tasks are supported:

- `sentiment-analysis` / `text-classification`
- `token-classification` / `ner`
- `question-answering`
- `fill-mask`
- `summarization`
- `translation`
- `text-generation`
- `text2text-generation`
- `zero-shot-classification`
- `audio-classification`
- `automatic-speech-recognition` / `asr`
- `text-to-audio` / `text-to-speech`
- `image-to-text`
- `image-classification`
- `image-segmentation`
- `zero-shot-image-classification`
- `object-detection`
- `zero-shot-object-detection`
- `document-question-answering`
- `image-to-image`
- `depth-estimation`
- `feature-extraction` / `embeddings`
- `image-feature-extraction`

## API Reference

### `usePipeline(options)`

React hook for loading and using a Transformers.js pipeline.

#### Parameters

- `options.task` (string, required): The task type (e.g., 'sentiment-analysis')
- `options.model` (string, optional): The model to use (defaults to task-specific model)
- `options.*`: Additional options passed to the Transformers.js `pipeline()` function

#### Returns

- `pipeline`: The loaded pipeline instance (or null if loading/error)
- `loading`: Boolean indicating if the model is loading
- `error`: Error object if loading failed (or null)
- `run(input, options?)`: Async function to run inference

## License

Apache-2.0

## Links

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [GitHub Repository](https://github.com/huggingface/transformers.js)
- [Hugging Face](https://huggingface.co)
