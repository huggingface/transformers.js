<!-- DO NOT EDIT: generated from src/**/*.js by docs/scripts/generate-skill.js -->

# Tasks

Runnable recipes for every task exposed through the `pipeline()` API, grouped by modality.
Each section is pulled from the pipeline class's JSDoc, so the examples stay in sync with the library.

## Contents

**Audio** — [`audio-classification`](#audio-classification) · [`zero-shot-audio-classification`](#zero-shot-audio-classification) · [`automatic-speech-recognition`](#automatic-speech-recognition) · [`text-to-audio`](#text-to-audio)

**Vision** — [`image-to-text`](#image-to-text) · [`image-classification`](#image-classification) · [`image-segmentation`](#image-segmentation) · [`background-removal`](#background-removal) · [`zero-shot-image-classification`](#zero-shot-image-classification) · [`object-detection`](#object-detection) · [`zero-shot-object-detection`](#zero-shot-object-detection) · [`document-question-answering`](#document-question-answering) · [`image-to-image`](#image-to-image) · [`depth-estimation`](#depth-estimation) · [`image-feature-extraction`](#image-feature-extraction)

**Text** — [`text-classification`](#text-classification) · [`token-classification`](#token-classification) · [`question-answering`](#question-answering) · [`fill-mask`](#fill-mask) · [`summarization`](#summarization) · [`translation`](#translation) · [`text2text-generation`](#text2text-generation) · [`text-generation`](#text-generation) · [`zero-shot-classification`](#zero-shot-classification)

**Embeddings** — [`feature-extraction`](#feature-extraction)

## Audio

### `audio-classification`

**Default model:** `Xenova/wav2vec2-base-superb-ks`

Audio classification pipeline using any `AutoModelForAudioClassification`.
This pipeline predicts the class of a raw waveform or an audio file.

**Example:** Perform audio classification with `Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('audio-classification', 'Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech');
const audio = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await classifier(audio);
// [
//   { label: 'male', score: 0.9981542229652405 },
//   { label: 'female', score: 0.001845747814513743 }
// ]
```

**Example:** Perform audio classification with `Xenova/ast-finetuned-audioset-10-10-0.4593` and return top 4 results.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('audio-classification', 'Xenova/ast-finetuned-audioset-10-10-0.4593');
const audio = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cat_meow.wav';
const output = await classifier(audio, { top_k: 4 });
// [
//   { label: 'Meow', score: 0.5617874264717102 },
//   { label: 'Cat', score: 0.22365376353263855 },
//   { label: 'Domestic animals, pets', score: 0.1141069084405899 },
//   { label: 'Animal', score: 0.08985692262649536 },
// ]
```

### `zero-shot-audio-classification`

**Default model:** `Xenova/clap-htsat-unfused`

Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you
provide an audio and a set of `candidate_labels`.

**Example**: Perform zero-shot audio classification with `Xenova/clap-htsat-unfused`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('zero-shot-audio-classification', 'Xenova/clap-htsat-unfused');
const audio = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/dog_barking.wav';
const candidate_labels = ['dog', 'vaccum cleaner'];
const scores = await classifier(audio, candidate_labels);
// [
//   { score: 0.9993992447853088, label: 'dog' },
//   { score: 0.0006007603369653225, label: 'vaccum cleaner' }
// ]
```

### `automatic-speech-recognition`

**Default model:** `Xenova/whisper-tiny.en`
**Aliases:** `asr`

Pipeline that aims at extracting spoken text contained within some audio.

**Example:** Transcribe English.
```javascript
import { pipeline } from '@huggingface/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await transcriber(url);
// { text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country." }
```

**Example:** Transcribe English w/ timestamps.
```javascript
import { pipeline } from '@huggingface/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await transcriber(url, { return_timestamps: true });
// {
//   text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
//   chunks: [
//     { timestamp: [0, 8],  text: " And so my fellow Americans ask not what your country can do for you" }
//     { timestamp: [8, 11], text: " ask what you can do for your country." }
//   ]
// }
```

**Example:** Transcribe English w/ word-level timestamps.
```javascript
import { pipeline } from '@huggingface/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await transcriber(url, { return_timestamps: 'word' });
// {
//   "text": " And so my fellow Americans ask not what your country can do for you ask what you can do for your country.",
//   "chunks": [
//     { "text": " And", "timestamp": [0, 0.78] },
//     { "text": " so", "timestamp": [0.78, 1.06] },
//     { "text": " my", "timestamp": [1.06, 1.46] },
//     ...
//     { "text": " for", "timestamp": [9.72, 9.92] },
//     { "text": " your", "timestamp": [9.92, 10.22] },
//     { "text": " country.", "timestamp": [10.22, 13.5] }
//   ]
// }
```

**Example:** Transcribe French.
```javascript
import { pipeline } from '@huggingface/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3';
const output = await transcriber(url, { language: 'french', task: 'transcribe' });
// { text: " J'adore, j'aime, je n'aime pas, je déteste." }
```

**Example:** Translate French to English.
```javascript
import { pipeline } from '@huggingface/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3';
const output = await transcriber(url, { language: 'french', task: 'translate' });
// { text: " I love, I like, I don't like, I hate." }
```

**Example:** Transcribe/translate audio longer than 30 seconds.
```javascript
import { pipeline } from '@huggingface/transformers';

const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/ted_60.wav';
const output = await transcriber(url, { chunk_length_s: 30, stride_length_s: 5 });
// { text: " So in college, I was a government major, which means [...] So I'd start off light and I'd bump it up" }
```

### `text-to-audio`

**Default model:** `onnx-community/Supertonic-TTS-ONNX`
**Aliases:** `text-to-speech`

Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`.
This pipeline generates an audio file from an input text and optional other conditional inputs.

**Example:** Generate audio from text with `onnx-community/Supertonic-TTS-ONNX`.
```javascript
import { pipeline } from '@huggingface/transformers';

const synthesizer = await pipeline('text-to-speech', 'onnx-community/Supertonic-TTS-ONNX');
const speaker_embeddings = 'https://huggingface.co/onnx-community/Supertonic-TTS-ONNX/resolve/main/voices/F1.bin';
const output = await synthesizer('Hello there, how are you doing?', { speaker_embeddings });
// RawAudio {
//   audio: Float32Array(95232) [-0.000482565927086398, -0.0004853440332226455, ...],
//   sampling_rate: 44100
// }

// Optional: Save the audio to a .wav file or Blob
await output.save('output.wav'); // You can also use `output.toBlob()` to access the audio as a Blob
```

**Example:** Multilingual speech generation with `Xenova/mms-tts-fra`. See [here](https://huggingface.co/models?pipeline_tag=text-to-speech&other=vits&sort=trending) for the full list of available languages (1107).
```javascript
import { pipeline } from '@huggingface/transformers';

const synthesizer = await pipeline('text-to-speech', 'Xenova/mms-tts-fra');
const output = await synthesizer('Bonjour');
// RawAudio {
//   audio: Float32Array(23808) [-0.00037693005288019776, 0.0003325853613205254, ...],
//   sampling_rate: 16000
// }
```

## Vision

### `image-to-text`

**Default model:** `Xenova/vit-gpt2-image-captioning`

Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

**Example:** Generate a caption for an image w/ `Xenova/vit-gpt2-image-captioning`.
```javascript
import { pipeline } from '@huggingface/transformers';

const captioner = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await captioner(url);
// [{ generated_text: 'a cat laying on a couch with another cat' }]
```

**Example:** Optical Character Recognition (OCR) w/ `Xenova/trocr-small-handwritten`.
```javascript
import { pipeline } from '@huggingface/transformers';

const captioner = await pipeline('image-to-text', 'Xenova/trocr-small-handwritten');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/handwriting.jpg';
const output = await captioner(url);
// [{ generated_text: 'Mr. Brown commented icily.' }]
```

### `image-classification`

**Default model:** `Xenova/vit-base-patch16-224`

Image classification pipeline using any `AutoModelForImageClassification`.
This pipeline predicts the class of an image.

**Example:** Classify an image.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url);
// [
//   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
// ]
```

**Example:** Classify an image and return top `n` classes.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url, { top_k: 3 });
// [
//   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
//   { label: 'tiger cat', score: 0.3634825646877289 },
//   { label: 'lion, king of beasts, Panthera leo', score: 0.00045060308184474707 },
// ]
```

**Example:** Classify an image and return all classes.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url, { top_k: 0 });
// [
//   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
//   { label: 'tiger cat', score: 0.3634825646877289 },
//   { label: 'lion, king of beasts, Panthera leo', score: 0.00045060308184474707 },
//   { label: 'jaguar, panther, Panthera onca, Felis onca', score: 0.00035465499968267977 },
//   ...
// ]
```

### `image-segmentation`

**Default model:** `Xenova/detr-resnet-50-panoptic`

Image segmentation pipeline using any `AutoModelForXXXSegmentation`.
This pipeline predicts masks of objects and their classes.

**Example:** Perform image segmentation with `Xenova/detr-resnet-50-panoptic`.
```javascript
import { pipeline } from '@huggingface/transformers';

const segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await segmenter(url);
// [
//   { label: 'remote', score: 0.9984649419784546, mask: RawImage { ... } },
//   { label: 'cat', score: 0.9994316101074219, mask: RawImage { ... } }
// ]
```

### `background-removal`

**Default model:** `Xenova/modnet`

Background removal pipeline using certain `AutoModelForXXXSegmentation`.
This pipeline removes the backgrounds of images.

**Example:** Perform background removal with `Xenova/modnet`.
```javascript
import { pipeline } from '@huggingface/transformers';

const segmenter = await pipeline('background-removal', 'Xenova/modnet');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman_small.jpg';
const output = await segmenter(url);
// RawImage { data: Uint8ClampedArray(648000) [ ... ], width: 360, height: 450, channels: 4 }
```

### `zero-shot-image-classification`

**Default model:** `Xenova/clip-vit-base-patch32`

Zero shot image classification pipeline. This pipeline predicts the class of
an image when you provide an image and a set of `candidate_labels`.

**Example:** Zero shot image classification w/ `Xenova/clip-vit-base-patch32`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url, ['tiger', 'horse', 'dog']);
// [
//   { score: 0.9993917942047119, label: 'tiger' },
//   { score: 0.0003519294841680676, label: 'horse' },
//   { score: 0.0002562698791734874, label: 'dog' }
// ]
```

### `object-detection`

**Default model:** `Xenova/detr-resnet-50`

Object detection pipeline using any `AutoModelForObjectDetection`.
This pipeline predicts bounding boxes of objects and their classes.

**Example:** Run object-detection with `Xenova/detr-resnet-50`.
```javascript
import { pipeline } from '@huggingface/transformers';

const detector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await detector(img, { threshold: 0.9 });
// [{
//   score: 0.9976370930671692,
//   label: "remote",
//   box: { xmin: 31, ymin: 68, xmax: 190, ymax: 118 }
// },
// ...
// {
//   score: 0.9984092116355896,
//   label: "cat",
//   box: { xmin: 331, ymin: 19, xmax: 649, ymax: 371 }
// }]
```

### `zero-shot-object-detection`

**Default model:** `Xenova/owlvit-base-patch32`

Zero-shot object detection pipeline. This pipeline predicts bounding boxes of
objects when you provide an image and a set of `candidate_labels`.

**Example:** Zero-shot object detection w/ `Xenova/owlvit-base-patch32`.
```javascript
import { pipeline } from '@huggingface/transformers';

const detector = await pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/astronaut.png';
const candidate_labels = ['human face', 'rocket', 'helmet', 'american flag'];
const output = await detector(url, candidate_labels);
// [
//   {
//     score: 0.24392342567443848,
//     label: 'human face',
//     box: { xmin: 180, ymin: 67, xmax: 274, ymax: 175 }
//   },
//   {
//     score: 0.15129457414150238,
//     label: 'american flag',
//     box: { xmin: 0, ymin: 4, xmax: 106, ymax: 513 }
//   },
//   {
//     score: 0.13649864494800568,
//     label: 'helmet',
//     box: { xmin: 277, ymin: 337, xmax: 511, ymax: 511 }
//   },
//   {
//     score: 0.10262022167444229,
//     label: 'rocket',
//     box: { xmin: 352, ymin: -1, xmax: 463, ymax: 287 }
//   }
// ]
```

**Example:** Zero-shot object detection w/ `Xenova/owlvit-base-patch32` (returning top 4 matches and setting a threshold).
```javascript
import { pipeline } from '@huggingface/transformers';

const detector = await pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/beach.png';
const candidate_labels = ['hat', 'book', 'sunglasses', 'camera'];
const output = await detector(url, candidate_labels, { top_k: 4, threshold: 0.05 });
// [
//   {
//     score: 0.1606510728597641,
//     label: 'sunglasses',
//     box: { xmin: 347, ymin: 229, xmax: 429, ymax: 264 }
//   },
//   {
//     score: 0.08935828506946564,
//     label: 'hat',
//     box: { xmin: 38, ymin: 174, xmax: 258, ymax: 364 }
//   },
//   {
//     score: 0.08530698716640472,
//     label: 'camera',
//     box: { xmin: 187, ymin: 350, xmax: 260, ymax: 411 }
//   },
//   {
//     score: 0.08349756896495819,
//     label: 'book',
//     box: { xmin: 261, ymin: 280, xmax: 494, ymax: 425 }
//   }
// ]
```

### `document-question-answering`

**Default model:** `Xenova/donut-base-finetuned-docvqa`

Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`.
The inputs/outputs are similar to the (extractive) question answering pipeline; however,
the pipeline takes an image (and optional OCR'd words/boxes) as input instead of text context.

**Example:** Answer questions about a document with `Xenova/donut-base-finetuned-docvqa`.
```javascript
import { pipeline } from '@huggingface/transformers';

const qa_pipeline = await pipeline('document-question-answering', 'Xenova/donut-base-finetuned-docvqa');
const image = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png';
const question = 'What is the invoice number?';
const output = await qa_pipeline(image, question);
// [{ answer: 'us-001' }]
```

### `image-to-image`

**Default model:** `Xenova/swin2SR-classical-sr-x2-64`

Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous image input.

**Example:** Super-resolution w/ `Xenova/swin2SR-classical-sr-x2-64`
```javascript
import { pipeline } from '@huggingface/transformers';

const upscaler = await pipeline('image-to-image', 'Xenova/swin2SR-classical-sr-x2-64');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/butterfly.jpg';
const output = await upscaler(url);
// RawImage {
//   data: Uint8Array(786432) [ 41, 31, 24,  43, ... ],
//   width: 512,
//   height: 512,
//   channels: 3
// }
```

### `depth-estimation`

**Default model:** `onnx-community/depth-anything-v2-small`

Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

**Example:** Depth estimation w/ `onnx-community/depth-anything-v2-small`
```javascript
import { pipeline } from '@huggingface/transformers';

const depth_estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small');
const image = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await depth_estimator(image);
// {
//   predicted_depth: Tensor {
//     dims: [ 480, 640 ],
//     type: 'float32',
//     data: Float32Array(307200) [ 2.6300313472747803, 2.5856235027313232, 2.620532751083374, ... ],
//     size: 307200
//   },
//   depth: RawImage {
//     data: Uint8Array(307200) [ 106, 104, 106, ... ],
//     width: 640,
//     height: 480,
//     channels: 1
//   }
// }
```

### `image-feature-extraction`

**Default model:** `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`

Image feature extraction pipeline using no model head. This pipeline extracts the hidden
states from the base transformer, which can be used as features in downstream tasks.

**Example:** Perform image feature extraction with `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`.
```javascript
import { pipeline } from '@huggingface/transformers';

const image_feature_extractor = await pipeline('image-feature-extraction', 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX');
const image = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png';
const features = await image_feature_extractor(image);
// Tensor {
//   dims: [ 1, 201, 384 ],
//   type: 'float32',
//   data: Float32Array(77184) [ ... ],
//   size: 77184
// }
```

**Example:** Compute image embeddings with `Xenova/clip-vit-base-patch32`.
```javascript
import { pipeline } from '@huggingface/transformers';

const image_feature_extractor = await pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32');
const image = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png';
const features = await image_feature_extractor(image);
// Tensor {
//   dims: [ 1, 512 ],
//   type: 'float32',
//   data: Float32Array(512) [ ... ],
//   size: 512
// }
```

## Text

### `text-classification`

**Default model:** `Xenova/distilbert-base-uncased-finetuned-sst-2-english`
**Aliases:** `sentiment-analysis`

Text classification pipeline using any `ModelForSequenceClassification`.

**Example:** Sentiment-analysis w/ `Xenova/distilbert-base-uncased-finetuned-sst-2-english`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
const output = await classifier('I love transformers!');
// [{ label: 'POSITIVE', score: 0.999788761138916 }]
```

**Example:** Multilingual sentiment-analysis w/ `Xenova/bert-base-multilingual-uncased-sentiment` (and return top 5 classes).
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('sentiment-analysis', 'Xenova/bert-base-multilingual-uncased-sentiment');
const output = await classifier('Le meilleur film de tous les temps.', { top_k: 5 });
// [
//   { label: '5 stars', score: 0.9610759615898132 },
//   { label: '4 stars', score: 0.03323351591825485 },
//   { label: '3 stars', score: 0.0036155181005597115 },
//   { label: '1 star', score: 0.0011325967498123646 },
//   { label: '2 stars', score: 0.0009423971059732139 }
// ]
```

**Example:** Toxic comment classification w/ `Xenova/toxic-bert` (and return all classes).
```javascript
const classifier = await pipeline('text-classification', 'Xenova/toxic-bert');
const output = await classifier('I hate you!', { top_k: null });
// [
//   { label: 'toxic', score: 0.9593140482902527 },
//   { label: 'insult', score: 0.16187334060668945 },
//   { label: 'obscene', score: 0.03452680632472038 },
//   { label: 'identity_hate', score: 0.0223250575363636 },
//   { label: 'threat', score: 0.019197041168808937 },
//   { label: 'severe_toxic', score: 0.005651099607348442 }
// ]
```

### `token-classification`

**Default model:** `Xenova/bert-base-multilingual-cased-ner-hrl`
**Aliases:** `ner`

Named Entity Recognition pipeline using any `ModelForTokenClassification`.

**Example:** Perform named entity recognition with `Xenova/bert-base-NER`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
const output = await classifier('My name is Sarah and I live in London');
// [
//   { entity: 'B-PER', score: 0.9980202913284302, index: 4, word: 'Sarah' },
//   { entity: 'B-LOC', score: 0.9994474053382874, index: 9, word: 'London' }
// ]
```

**Example:** Perform named entity recognition with `Xenova/bert-base-NER` (and return all labels).
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
const output = await classifier('Sarah lives in the United States of America', { ignore_labels: [] });
// [
//   { entity: 'B-PER', score: 0.9966587424278259, index: 1, word: 'Sarah' },
//   { entity: 'O', score: 0.9987385869026184, index: 2, word: 'lives' },
//   { entity: 'O', score: 0.9990072846412659, index: 3, word: 'in' },
//   { entity: 'O', score: 0.9988298416137695, index: 4, word: 'the' },
//   { entity: 'B-LOC', score: 0.9995510578155518, index: 5, word: 'United' },
//   { entity: 'I-LOC', score: 0.9990395307540894, index: 6, word: 'States' },
//   { entity: 'I-LOC', score: 0.9986724853515625, index: 7, word: 'of' },
//   { entity: 'I-LOC', score: 0.9975294470787048, index: 8, word: 'America' }
// ]
```

**Example:** Group adjacent BIO/BIOES tokens into entity spans using `aggregation_strategy: "simple"`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
const output = await classifier('My name is Sarah and I live in London', { aggregation_strategy: 'simple' });
// [
//   { entity_group: 'PER', score: 0.9985477924346924, word: 'Sarah' },
//   { entity_group: 'LOC', score: 0.999621570110321, word: 'London' }
// ]
```

### `question-answering`

**Default model:** `Xenova/distilbert-base-cased-distilled-squad`

Question Answering pipeline using any `ModelForQuestionAnswering`.

**Example:** Run question answering with `Xenova/distilbert-base-uncased-distilled-squad`.
```javascript
import { pipeline } from '@huggingface/transformers';

const answerer = await pipeline('question-answering', 'Xenova/distilbert-base-uncased-distilled-squad');
const question = 'Who was Jim Henson?';
const context = 'Jim Henson was a nice puppet.';
const output = await answerer(question, context);
// {
//   answer: "a nice puppet",
//   score: 0.5768911502526741
// }
```

### `fill-mask`

**Default model:** `onnx-community/ettin-encoder-32m-ONNX`

Masked language modeling prediction pipeline using any `ModelWithLMHead`.

**Example:** Perform masked language modelling (a.k.a. "fill-mask") with `onnx-community/ettin-encoder-32m-ONNX`.
```javascript
import { pipeline } from '@huggingface/transformers';

const unmasker = await pipeline('fill-mask', 'onnx-community/ettin-encoder-32m-ONNX');
const output = await unmasker('The capital of France is [MASK].');
// [
//   { score: 0.5151872038841248, token: 7785, token_str: ' Paris', sequence: 'The capital of France is Paris.' },
//   { score: 0.033725105226039886, token: 42268, token_str: ' Lyon', sequence: 'The capital of France is Lyon.' },
//   { score: 0.031234024092555046, token: 23397, token_str: ' Nancy', sequence: 'The capital of France is Nancy.' },
//   { score: 0.02075139433145523, token: 30167, token_str: ' Brussels', sequence: 'The capital of France is Brussels.' },
//   { score: 0.018962178379297256, token: 31955, token_str: ' Geneva', sequence: 'The capital of France is Geneva.' }
// ]
```

**Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-uncased`.
```javascript
import { pipeline } from '@huggingface/transformers';

const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
const output = await unmasker('The goal of life is [MASK].');
// [
//   { score: 0.11368396878242493, sequence: "The goal of life is survival.", token: 8115, token_str: "survival" },
//   { score: 0.053510840982198715, sequence: "The goal of life is love.", token: 1567, token_str: "love" },
//   { score: 0.05041185021400452, sequence: "The goal of life is happiness.", token: 9266, token_str: "happiness" },
//   { score: 0.033218126744031906, sequence: "The goal of life is freedom.", token: 4438, token_str: "freedom" },
//   { score: 0.03301157429814339, sequence: "The goal of life is success.", token: 2244, token_str: "success" },
// ]
```

**Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-cased` (and return top result).
```javascript
import { pipeline } from '@huggingface/transformers';

const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
const output = await unmasker('The Milky Way is a [MASK] galaxy.', { top_k: 1 });
// [{ score: 0.5982972383499146, sequence: "The Milky Way is a spiral galaxy.", token: 14061, token_str: "spiral" }]
```

### `summarization`

**Default model:** `Xenova/distilbart-cnn-6-6`

A pipeline for summarization tasks, inheriting from Text2TextGenerationPipeline.

**Example:** Summarization w/ `Xenova/distilbart-cnn-6-6`.
```javascript
import { pipeline } from '@huggingface/transformers';

const summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
const text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, ' +
  'and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. ' +
  'During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest ' +
  'man-made structure in the world, a title it held for 41 years until the Chrysler Building in New ' +
  'York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to ' +
  'the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the ' +
  'Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second ' +
  'tallest free-standing structure in France after the Millau Viaduct.';
const output = await summarizer(text, {
  max_new_tokens: 100,
});
// [{ summary_text: ' The Eiffel Tower is about the same height as an 81-storey building and the tallest structure in Paris. It is the second tallest free-standing structure in France after the Millau Viaduct.' }]
```

### `translation`

**Default model:** `Xenova/t5-small`

Translates text from one language to another.

**Example:** Multilingual translation w/ `Xenova/nllb-200-distilled-600M`.

See [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
for the full list of languages and their corresponding codes.

```javascript
import { pipeline } from '@huggingface/transformers';

const translator = await pipeline('translation', 'Xenova/nllb-200-distilled-600M');
const output = await translator('जीवन एक चॉकलेट बॉक्स की तरह है।', {
  src_lang: 'hin_Deva', // Hindi
  tgt_lang: 'fra_Latn', // French
});
// [{ translation_text: 'La vie est comme une boîte à chocolat.' }]
```

**Example:** Multilingual translation w/ `Xenova/m2m100_418M`.

See [here](https://huggingface.co/facebook/m2m100_418M#languages-covered)
for the full list of languages and their corresponding codes.

```javascript
import { pipeline } from '@huggingface/transformers';

const translator = await pipeline('translation', 'Xenova/m2m100_418M');
const output = await translator('生活就像一盒巧克力。', {
  src_lang: 'zh', // Chinese
  tgt_lang: 'en', // English
});
// [{ translation_text: 'Life is like a box of chocolate.' }]
```

**Example:** Multilingual translation w/ `Xenova/mbart-large-50-many-to-many-mmt`.

See [here](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered)
for the full list of languages and their corresponding codes.

```javascript
import { pipeline } from '@huggingface/transformers';

const translator = await pipeline('translation', 'Xenova/mbart-large-50-many-to-many-mmt');
const output = await translator('संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है', {
  src_lang: 'hi_IN', // Hindi
  tgt_lang: 'fr_XX', // French
});
// [{ translation_text: 'Le chef des Nations affirme qu 'il n 'y a military solution in Syria.' }]
```

### `text2text-generation`

**Default model:** `Xenova/flan-t5-small`

Text2TextGenerationPipeline class for generating text using a model that performs text-to-text generation tasks.

**Example:** Text-to-text generation w/ `Xenova/LaMini-Flan-T5-783M`.
```javascript
import { pipeline } from '@huggingface/transformers';

const generator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M');
const output = await generator('how can I become more healthy?', {
  max_new_tokens: 100,
});
// [{ generated_text: "To become more healthy, you can: 1. Eat a balanced diet with plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. 2. Stay hydrated by drinking plenty of water. 3. Get enough sleep and manage stress levels. 4. Avoid smoking and excessive alcohol consumption. 5. Regularly exercise and maintain a healthy weight. 6. Practice good hygiene and sanitation. 7. Seek medical attention if you experience any health issues." }]
```

### `text-generation`

**Default model:** `onnx-community/Qwen3-0.6B-ONNX`

Language generation pipeline using any `ModelWithLMHead` or `ModelForCausalLM`.
This pipeline predicts the words that will follow a specified text prompt.
For the full list of generation parameters, see [`GenerationConfig`](https://huggingface.co/docs/transformers.js/api/generation/configuration_utils#generationconfig).

**Example:** Text generation with `HuggingFaceTB/SmolLM2-135M` (default settings).
```javascript
import { pipeline } from '@huggingface/transformers';

const generator = await pipeline('text-generation', 'onnx-community/SmolLM2-135M-ONNX');
const text = 'Once upon a time,';
const output = await generator(text, { max_new_tokens: 8 });
// [{ generated_text: 'Once upon a time, there was a little girl named Lily.' }]
```

**Example:** Chat completion with `onnx-community/Qwen3-0.6B-ONNX`.
```javascript
import { pipeline, TextStreamer } from '@huggingface/transformers';

// Create a text generation pipeline
const generator = await pipeline(
  'text-generation',
  'onnx-community/Qwen3-0.6B-ONNX',
  { dtype: 'q4f16' },
);

// Define the list of messages
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write me a poem about Machine Learning.' },
];

// Generate a response
const output = await generator(messages, {
  max_new_tokens: 512,
  do_sample: false,
  streamer: new TextStreamer(generator.tokenizer, { skip_prompt: true, skip_special_tokens: true }),
});
console.log(output[0].generated_text.at(-1)?.content);
```

### `zero-shot-classification`

**Default model:** `Xenova/distilbert-base-uncased-mnli`

NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification`
trained on NLI (natural language inference) tasks. Equivalent of `text-classification`
pipelines, but these models don't require a hardcoded number of potential classes, they
can be chosen at runtime. It usually means it's slower but it is **much** more flexible.

**Example:** Zero shot classification with `Xenova/mobilebert-uncased-mnli`.
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('zero-shot-classification', 'Xenova/mobilebert-uncased-mnli');
const text = 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.';
const labels = [ 'mobile', 'billing', 'website', 'account access' ];
const output = await classifier(text, labels);
// {
//   sequence: 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.',
//   labels: [ 'mobile', 'website', 'billing', 'account access' ],
//   scores: [ 0.5562091040482018, 0.1843621307860853, 0.13942646639336376, 0.12000229877234923 ]
// }
```

**Example:** Zero shot classification with `Xenova/nli-deberta-v3-xsmall` (multi-label).
```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('zero-shot-classification', 'Xenova/nli-deberta-v3-xsmall');
const text = 'I have a problem with my iphone that needs to be resolved asap!';
const labels = [ 'urgent', 'not urgent', 'phone', 'tablet', 'computer' ];
const output = await classifier(text, labels, { multi_label: true });
// {
//   sequence: 'I have a problem with my iphone that needs to be resolved asap!',
//   labels: [ 'urgent', 'phone', 'computer', 'tablet', 'not urgent' ],
//   scores: [ 0.9958870956360275, 0.9923963400697035, 0.002333537946160235, 0.0015134138567598765, 0.0010699384208377163 ]
// }
```

## Embeddings

### `feature-extraction`

**Default model:** `onnx-community/all-MiniLM-L6-v2-ONNX`
**Aliases:** `embeddings`

Feature extraction pipeline using no model head. This pipeline extracts the hidden
states from the base transformer, which can be used as features in downstream tasks.

**Example:** Run feature extraction using `onnx-community/all-MiniLM-L6-v2-ONNX` (without pooling or normalization).
```javascript
import { pipeline } from '@huggingface/transformers';

const extractor = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
const output = await extractor('This is a simple test.');
// Tensor {
//   type: 'float32',
//   data: Float32Array [0.2157987803220749, -0.09140099585056305, ...],
//   dims: [1, 8, 384]
// }

// You can convert this Tensor to a nested JavaScript array using `.tolist()`:
console.log(output.tolist());
```

**Example:** Run feature extraction using `onnx-community/all-MiniLM-L6-v2-ONNX` (with pooling and normalization).
```javascript
import { pipeline } from '@huggingface/transformers';

const extractor = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
// Tensor {
//   type: 'float32',
//   data: Float32Array [0.09528215229511261, -0.024730168282985687, ...],
//   dims: [1, 384]
// }

// You can convert this Tensor to a nested JavaScript array using `.tolist()`:
console.log(output.tolist());
```

**Example:** Run feature extraction using `onnx-community/all-MiniLM-L6-v2-ONNX` models (with pooling and binary quantization).
```javascript
const extractor = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX');
const output = await extractor('This is a simple test.', { pooling: 'mean', quantize: true, precision: 'binary' });
// Tensor {
//   type: 'int8',
//   data: Int8Array [49, 108, 25, ...],
//   dims: [1, 48]
// }

// You can convert this Tensor to a nested JavaScript array using `.tolist()`:
console.log(output.tolist());
```
