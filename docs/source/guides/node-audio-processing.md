
# Server-side Audio Processing in Node.js

A major benefit of writing code for the web is that you can access the multitude of APIs that are available in modern browsers. Unfortunately, when writing server-side code, we are not afforded such luxury, so we have to find another way. In this tutorial, we will design a simple Node.js application that uses Transformers.js for speech recognition with [Whisper](https://huggingface.co/Xenova/whisper-tiny.en), and in the process, learn how to process audio on the server.

The main problem we need to solve is that the [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) is not available in Node.js, meaning we can't use the [`AudioContext`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext) class to process audio. So, we will need to install third-party libraries to obtain the raw audio data. For this example, we will only consider `.wav` files, but the same principles apply to other audio formats.

<Tip>

This tutorial will be written as an ES module, but you can easily adapt it to use CommonJS instead. For more information, see the [node tutorial](https://huggingface.co/docs/transformers.js/tutorials/node).

</Tip>


**Useful links:**
- [Source code](https://github.com/huggingface/transformers.js/tree/main/examples/node-audio-processing)
- [Documentation](https://huggingface.co/docs/transformers.js)


## Prerequisites

- [Node.js](https://nodejs.org/en/) version 18+
- [npm](https://www.npmjs.com/) version 9+



## Getting started

Let's start by creating a new Node.js project and installing Transformers.js via [NPM](https://www.npmjs.com/package/@huggingface/transformers):

```bash
npm init -y
npm i @huggingface/transformers
```

<Tip>

Remember to add `"type": "module"` to your `package.json` to indicate that your project uses ECMAScript modules.

</Tip>


Next, let's install the [`wavefile`](https://www.npmjs.com/package/wavefile) package, which we will use for loading `.wav` files:

```bash
npm i wavefile
```


## Creating the application

Start by creating a new file called `index.js`, which will be the entry point for our application. Let's also import the necessary modules:

```js
import { pipeline } from '@huggingface/transformers';
import wavefile from 'wavefile';
```

For this tutorial, we will use the `Xenova/whisper-tiny.en` model, but feel free to choose one of the other whisper models from the [Hugging Face Hub](https://huggingface.co/models?library=transformers.js&search=whisper). Let's create our pipeline with:
```js
let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
```

Next, let's load an audio file and convert it to the format required by Transformers.js:
```js
// Load audio data
let url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
let buffer = Buffer.from(await fetch(url).then(x => x.arrayBuffer()))

// Read .wav file and convert it to required format
let wav = new wavefile.WaveFile(buffer);
wav.toBitDepth('32f'); // Pipeline expects input as a Float32Array
wav.toSampleRate(16000); // Whisper expects audio with a sampling rate of 16000
let audioData = wav.getSamples();
if (Array.isArray(audioData)) {
  if (audioData.length > 1) {
    const SCALING_FACTOR = Math.sqrt(2);

    // Merge channels (into first channel to save memory)
    for (let i = 0; i < audioData[0].length; ++i) {
      audioData[0][i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
    }
  }

  // Select first channel
  audioData = audioData[0];
}
```

Finally, let's run the model and measure execution duration.
```js
let start = performance.now();
let output = await transcriber(audioData);
let end = performance.now();
console.log(`Execution duration: ${(end - start) / 1000} seconds`);
console.log(output);
```

You can now run the application with `node index.js`. Note that when running the script for the first time, it may take a while to download and cache the model. Subsequent requests will use the cached model, and model loading will be much faster.

You should see output similar to:
```
Execution duration: 0.6460317999720574 seconds
{
  text: ' And so my fellow Americans ask not what your country can do for you. Ask what you can do for your country.'
}
```


That's it! You've successfully created a Node.js application that uses Transformers.js for speech recognition with Whisper. You can now use this as a starting point for your own applications.

