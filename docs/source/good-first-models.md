# Good-First Models

This guide provides curated model recommendations for each task type, selected for their:
- **Popularity**: Widely used with strong community support
- **Performance**: Fast loading and inference times
- **WebGPU Compatibility**: GPU-accelerated in modern browsers

**Important:** These recommendations are designed for initial experimentation and learning. Many other models are available for each task. **You should evaluate and choose the best model for your specific use case, performance requirements, and constraints.**

## About the Model Recommendations

The models below are selected for their popularity and ease of use, making them ideal for initial experimentation. **This list does not cover all available models** - you should evaluate and select the best model for your specific use case and requirements.

## Audio-Classification

**Usage Example:** Perform audio classification

```javascript
const classifier = await pipeline('audio-classification', 'onnx-community/Musical-genres-Classification-Hubert-V1-ONNX');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await classifier(url);
```

### Recommended Models for First-Time Trials

#### [onnx-community/Musical-genres-Classification-Hubert-V1-ONNX](https://huggingface.co/onnx-community/Musical-genres-Classification-Hubert-V1-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 264.4ms | Inference: 136.5ms | Downloads: 6 | Likes: 2

#### [Xenova/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/Xenova/ast-finetuned-audioset-10-10-0.4593)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 2827.1ms | Inference: 544.1ms | Downloads: 36

#### [Xenova/ast-finetuned-speech-commands-v2](https://huggingface.co/Xenova/ast-finetuned-speech-commands-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 3254.1ms | Inference: 759.0ms | Downloads: 16 | Likes: 2

#### [onnx-community/Speech-Emotion-Classification-ONNX](https://huggingface.co/onnx-community/Speech-Emotion-Classification-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 3070.4ms | Inference: 574.9ms | Downloads: 10 | Likes: 2

#### [Xenova/wav2vec2-base-superb-ks](https://huggingface.co/Xenova/wav2vec2-base-superb-ks)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 3211.4ms | Inference: 584.7ms | Downloads: 17 | Likes: 1

---

## Automatic-Speech-Recognition

**Usage Example:** Transcribe audio from a URL

```javascript
const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await transcriber(url);
```

### Recommended Models for First-Time Trials

#### [Xenova/whisper-tiny.en](https://huggingface.co/Xenova/whisper-tiny.en)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1045.6ms | Inference: 1314.3ms | Downloads: 40.6k | Likes: 19

#### [Xenova/whisper-tiny](https://huggingface.co/Xenova/whisper-tiny)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1037.2ms | Inference: 1258.4ms | Downloads: 25.7k | Likes: 10

#### [onnx-community/moonshine-base-ONNX](https://huggingface.co/onnx-community/moonshine-base-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1906.3ms | Inference: 889.1ms | Downloads: 10.9k | Likes: 29

#### [onnx-community/whisper-base](https://huggingface.co/onnx-community/whisper-base)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1885.8ms | Inference: 878.9ms | Downloads: 16.0k | Likes: 22

#### [Xenova/whisper-base.en](https://huggingface.co/Xenova/whisper-base.en)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1591.3ms | Inference: 1941.5ms | Downloads: 28.0k | Likes: 1

---

## Depth-Estimation

**Usage Example:** Depth estimation

```javascript
const depth_estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const out = await depth_estimator(url);
```

### Recommended Models for First-Time Trials

#### [Xenova/depth-anything-small-hf](https://huggingface.co/Xenova/depth-anything-small-hf)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 414.3ms | Inference: 419.3ms | Downloads: 3.1k | Likes: 11

#### [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 450.0ms | Inference: 357.1ms | Downloads: 1.3k | Likes: 22

#### [onnx-community/metric3d-vit-small](https://huggingface.co/onnx-community/metric3d-vit-small)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 786.2ms | Inference: 515.7ms | Downloads: 21 | Likes: 2

#### [Xenova/glpn-nyu](https://huggingface.co/Xenova/glpn-nyu)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1454.5ms | Inference: 1076.9ms | Downloads: 8

#### [Xenova/glpn-kitti](https://huggingface.co/Xenova/glpn-kitti)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1451.4ms | Inference: 1083.3ms | Downloads: 8

---

## Feature-Extraction

**Usage Example:** Run feature extraction

```javascript
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
const output = await extractor('This is a simple test.');
```

### Recommended Models for First-Time Trials

#### [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 373.5ms | Inference: 143.9ms | Downloads: 425.1k | Likes: 89

#### [Xenova/bge-base-en-v1.5](https://huggingface.co/Xenova/bge-base-en-v1.5)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 3620.8ms | Inference: 816.4ms | Downloads: 1.1M | Likes: 8

#### [Xenova/gte-small](https://huggingface.co/Xenova/gte-small)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 588.7ms | Inference: 228.4ms | Downloads: 13.5k | Likes: 22

#### [onnx-community/bert_uncased_L-2_H-128_A-2-ONNX](https://huggingface.co/onnx-community/bert_uncased_L-2_H-128_A-2-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 110.8ms | Inference: 49.2ms | Downloads: 8

#### [Xenova/paraphrase-albert-base-v2](https://huggingface.co/Xenova/paraphrase-albert-base-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 328.8ms | Inference: 39.1ms | Downloads: 21

---

## Fill-Mask

**Usage Example:** Perform masked language modelling (a.k.a. "fill-mask")

```javascript
const unmasker = await pipeline('fill-mask', 'Xenova/distilbert-base-uncased');
const output = await unmasker('The goal of life is [MASK].');
```

### Recommended Models for First-Time Trials

#### [Xenova/distilbert-base-uncased](https://huggingface.co/Xenova/distilbert-base-uncased)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 769.8ms | Inference: 302.4ms | Downloads: 702 | Likes: 1

#### [Xenova/bert-base-uncased](https://huggingface.co/Xenova/bert-base-uncased)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 4432.7ms | Inference: 919.4ms | Downloads: 814 | Likes: 1

#### [Xenova/albert-base-v2](https://huggingface.co/Xenova/albert-base-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 324.9ms | Inference: 68.1ms | Downloads: 12 | Likes: 1

#### [Xenova/distilbert-base-cased](https://huggingface.co/Xenova/distilbert-base-cased)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 742.7ms | Inference: 301.1ms | Downloads: 15 | Likes: 1

#### [Xenova/albert-large-v2](https://huggingface.co/Xenova/albert-large-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 499.6ms | Inference: 78.8ms | Downloads: 11

---

## Image-Classification

**Usage Example:** Classify an image

```javascript
const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url);
```

### Recommended Models for First-Time Trials

#### [Xenova/vit-base-patch16-224](https://huggingface.co/Xenova/vit-base-patch16-224)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 2894.1ms | Inference: 476.3ms | Downloads: 5.3k | Likes: 2

#### [Xenova/facial_emotions_image_detection](https://huggingface.co/Xenova/facial_emotions_image_detection)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 2922.3ms | Inference: 479.3ms | Downloads: 375 | Likes: 6

#### [Xenova/resnet-18](https://huggingface.co/Xenova/resnet-18)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 152.1ms | Inference: 90.8ms | Downloads: 27 | Likes: 2

#### [Xenova/resnet-50](https://huggingface.co/Xenova/resnet-50)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 304.9ms | Inference: 160.5ms | Downloads: 677

#### [onnx-community/mobilenet_v2_1.0_224](https://huggingface.co/onnx-community/mobilenet_v2_1.0_224)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 90.7ms | Inference: 99.7ms | Downloads: 40 | Likes: 1

---

## Image-Feature-Extraction

**Usage Example:** Perform image feature extraction

```javascript
const image_feature_extractor = await pipeline('image-feature-extraction', 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX');
const url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png';
const features = await image_feature_extractor(url);
```

### Recommended Models for First-Time Trials

#### [onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 311.4ms | Inference: 352.4ms | Downloads: 1.4k | Likes: 13

#### [Xenova/dino-vits16](https://huggingface.co/Xenova/dino-vits16)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 321.7ms | Inference: 247.3ms | Downloads: 1.1k

#### [onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX-MHA-scores](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX-MHA-scores)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 342.8ms | Inference: 259.6ms | Downloads: 489 | Likes: 2

#### [onnx-community/dinov2-small-ONNX](https://huggingface.co/onnx-community/dinov2-small-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 311.5ms | Inference: 264.2ms | Downloads: 3 | Likes: 1

#### [onnx-community/dinov2-with-registers-small](https://huggingface.co/onnx-community/dinov2-with-registers-small)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 312.4ms | Inference: 270.6ms | Downloads: 3 | Likes: 1

---

## Image-Segmentation

**Usage Example:** Perform image segmentation

```javascript
const segmenter = await pipeline('image-segmentation', 'Xenova/segformer-b0-finetuned-ade-512-512');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await segmenter(url);
```

### Recommended Models for First-Time Trials

#### [Xenova/segformer-b0-finetuned-ade-512-512](https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 205.6ms | Inference: 1612.3ms | Downloads: 1.1M | Likes: 1

#### [Xenova/modnet](https://huggingface.co/Xenova/modnet)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 144.6ms | Inference: 286.9ms | Downloads: 7.4k | Likes: 65

#### [onnx-community/modnet-webnn](https://huggingface.co/onnx-community/modnet-webnn)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 147.5ms | Inference: 223.2ms | Downloads: 16 | Likes: 4

#### [Xenova/segformer_b0_clothes](https://huggingface.co/Xenova/segformer_b0_clothes)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 203.9ms | Inference: 632.4ms | Downloads: 114 | Likes: 2

#### [Xenova/segformer-b0-finetuned-cityscapes-1024-1024](https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-1024-1024)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 200.2ms | Inference: 519.0ms | Downloads: 7

---

## Image-To-Image

**Usage Example:**

```javascript
const processor = await pipeline('image-to-image', 'Xenova/2x_APISR_RRDB_GAN_generator-onnx');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await processor(url);
```

### Recommended Models for First-Time Trials

#### [Xenova/2x_APISR_RRDB_GAN_generator-onnx](https://huggingface.co/Xenova/2x_APISR_RRDB_GAN_generator-onnx)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 117.5ms | Inference: 596.3ms | Downloads: 18 | Likes: 2

#### [Xenova/4x_APISR_GRL_GAN_generator-onnx](https://huggingface.co/Xenova/4x_APISR_GRL_GAN_generator-onnx)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 2712.5ms | Inference: 3362.6ms | Downloads: 36 | Likes: 14

---

## Object-Detection

**Usage Example:** Run object-detection

```javascript
const detector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await detector(img, { threshold: 0.9 });
```

### Recommended Models for First-Time Trials

#### [Xenova/detr-resnet-50](https://huggingface.co/Xenova/detr-resnet-50)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 718.2ms | Inference: 625.4ms | Downloads: 16.5k | Likes: 16

#### [Xenova/yolos-tiny](https://huggingface.co/Xenova/yolos-tiny)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 190.1ms | Inference: 904.3ms | Downloads: 792 | Likes: 6

#### [onnx-community/rfdetr_base-ONNX](https://huggingface.co/onnx-community/rfdetr_base-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 470.4ms | Inference: 420.9ms | Downloads: 5 | Likes: 2

#### [Xenova/yolos-small-300](https://huggingface.co/Xenova/yolos-small-300)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 400.0ms | Inference: 1776.1ms | Downloads: 648

#### [onnx-community/rfdetr_medium-ONNX](https://huggingface.co/onnx-community/rfdetr_medium-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 578.1ms | Inference: 486.1ms | Downloads: 5 | Likes: 1

---

## Question-Answering

**Usage Example:** Run question answering

```javascript
const answerer = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad');
const question = 'Who was Jim Henson?';
const context = 'Jim Henson was a nice puppet.';
const output = await answerer(question, context);
```

### Recommended Models for First-Time Trials

#### [Xenova/distilbert-base-cased-distilled-squad](https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 746.8ms | Inference: 164.3ms | Downloads: 5.9k | Likes: 4

#### [onnx-community/rubert_tiny_qa_kontur-ONNX](https://huggingface.co/onnx-community/rubert_tiny_qa_kontur-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 411.4ms | Inference: 111.1ms | Downloads: 5

#### [Xenova/distilbert-base-uncased-distilled-squad](https://huggingface.co/Xenova/distilbert-base-uncased-distilled-squad)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 726.1ms | Inference: 204.5ms | Downloads: 914

#### [onnx-community/xlm-roberta-base-squad2-distilled-ONNX](https://huggingface.co/onnx-community/xlm-roberta-base-squad2-distilled-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 9046.6ms | Inference: 1201.1ms | Downloads: 21.9k

#### [onnx-community/mobilebert-uncased-squad-v2-ONNX](https://huggingface.co/onnx-community/mobilebert-uncased-squad-v2-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 465.3ms | Inference: 911.1ms | Downloads: 5

---

## Summarization

**Usage Example:** Summarization

```javascript
const generator = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
const text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, ' +
  'and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. ' +
  'During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest ' +
  'man-made structure in the world, a title it held for 41 years until the Chrysler Building in New ' +
  'York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to ' +
  'the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the ' +
  'Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second ' +
  'tallest free-standing structure in France after the Millau Viaduct.';
const output = await generator(text, {
  max_new_tokens: 100,
});
```

### Recommended Models for First-Time Trials

#### [Xenova/distilbart-cnn-6-6](https://huggingface.co/Xenova/distilbart-cnn-6-6)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 9311.4ms | Inference: 2527.9ms | Downloads: 4.2k | Likes: 9

#### [Xenova/distilbart-xsum-12-1](https://huggingface.co/Xenova/distilbart-xsum-12-1)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 8184.2ms | Inference: 1283.7ms | Downloads: 25

#### [Xenova/distilbart-xsum-6-6](https://huggingface.co/Xenova/distilbart-xsum-6-6)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 8895.0ms | Inference: 1519.4ms | Downloads: 28 | Likes: 1

#### [Xenova/distilbart-xsum-9-6](https://huggingface.co/Xenova/distilbart-xsum-9-6)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 10079.9ms | Inference: 1730.8ms | Downloads: 24

#### [Xenova/distilbart-cnn-12-3](https://huggingface.co/Xenova/distilbart-cnn-12-3)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 8878.6ms | Inference: 2479.6ms | Downloads: 29

---

## Text-Classification

**Usage Example:**

```javascript
const classifier = await pipeline('text-classification', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
const output = await classifier('I love transformers!');
```

### Recommended Models for First-Time Trials

#### [Xenova/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 662.4ms | Inference: 245.3ms | Downloads: 8.4k | Likes: 11

#### [Xenova/ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 365.1ms | Inference: 146.6ms | Downloads: 7.2k | Likes: 4

#### [Xenova/ms-marco-TinyBERT-L-2-v2](https://huggingface.co/Xenova/ms-marco-TinyBERT-L-2-v2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 115.1ms | Inference: 56.2ms | Downloads: 1.6k | Likes: 3

#### [Xenova/toxic-bert](https://huggingface.co/Xenova/toxic-bert)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 3789.0ms | Inference: 808.8ms | Downloads: 5.2k | Likes: 6

#### [onnx-community/language_detection-ONNX](https://huggingface.co/onnx-community/language_detection-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 338.1ms | Inference: 116.0ms | Downloads: 278 | Likes: 4

---

## Text-Generation

**Usage Example:** Text generation

```javascript
const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
const output = await generator('Once upon a time, there was', { max_new_tokens: 10 });
```

### Recommended Models for First-Time Trials

#### [Xenova/distilgpt2](https://huggingface.co/Xenova/distilgpt2)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 3207.5ms | Inference: 469.7ms | Downloads: 7.7k | Likes: 9

#### [Xenova/llama2.c-stories15M](https://huggingface.co/Xenova/llama2.c-stories15M)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 396.8ms | Inference: 251.4ms | Downloads: 1.9k | Likes: 7

#### [onnx-community/ettin-decoder-17m-ONNX](https://huggingface.co/onnx-community/ettin-decoder-17m-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 328.2ms | Inference: 221.9ms | Downloads: 3 | Likes: 1

#### [onnx-community/Lamina-extend-ONNX](https://huggingface.co/onnx-community/Lamina-extend-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 141.9ms | Inference: 190.0ms | Downloads: 6

#### [onnx-community/chess-llama-ONNX](https://huggingface.co/onnx-community/chess-llama-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 249.3ms | Inference: 197.2ms | Downloads: 3

---

## Token-Classification

**Usage Example:** Perform named entity recognition

```javascript
const classifier = await pipeline('token-classification', 'onnx-community/NeuroBERT-NER-ONNX');
const output = await classifier('My name is Sarah and I live in London');
```

### Recommended Models for First-Time Trials

#### [onnx-community/NeuroBERT-NER-ONNX](https://huggingface.co/onnx-community/NeuroBERT-NER-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 170.4ms | Inference: 61.6ms | Downloads: 7 | Likes: 2

#### [onnx-community/TinyBERT-finetuned-NER-ONNX](https://huggingface.co/onnx-community/TinyBERT-finetuned-NER-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 261.7ms | Inference: 79.9ms | Downloads: 8 | Likes: 1

#### [Xenova/esm2_t6_8M_UR50D_rna_binding_site_predictor](https://huggingface.co/Xenova/esm2_t6_8M_UR50D_rna_binding_site_predictor)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 279.3ms | Inference: 118.6ms | Downloads: 5 | Likes: 1

#### [onnx-community/distilbert-NER-ONNX](https://huggingface.co/onnx-community/distilbert-NER-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 682.0ms | Inference: 258.2ms | Downloads: 17 | Likes: 2

#### [onnx-community/small-e-czech-finetuned-ner-wikiann-ONNX](https://huggingface.co/onnx-community/small-e-czech-finetuned-ner-wikiann-ONNX)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 271.3ms | Inference: 193.4ms | Downloads: 6

---

## Translation

**Usage Example:** Multilingual translation

```javascript
const translator = await pipeline('translation', 'Xenova/opus-mt-en-es');
const output = await translator('Life is like a box of chocolate.', {
  src_lang: '...',
  tgt_lang: '...',
});
```

### Recommended Models for First-Time Trials

#### [Xenova/opus-mt-en-es](https://huggingface.co/Xenova/opus-mt-en-es)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1956.5ms | Inference: 617.7ms | Downloads: 2.0k | Likes: 3

#### [Xenova/opus-mt-zh-en](https://huggingface.co/Xenova/opus-mt-zh-en)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1964.7ms | Inference: 799.0ms | Downloads: 53 | Likes: 4

#### [Xenova/opus-mt-en-zh](https://huggingface.co/Xenova/opus-mt-en-zh)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1830.0ms | Inference: 481.5ms | Downloads: 490 | Likes: 2

#### [Xenova/opus-mt-en-fr](https://huggingface.co/Xenova/opus-mt-en-fr)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1786.1ms | Inference: 446.7ms | Downloads: 373 | Likes: 2

#### [Xenova/opus-mt-ar-en](https://huggingface.co/Xenova/opus-mt-ar-en)

**WebGPU Compatible:** ✅ Yes

**Metrics:** Load: 1756.6ms | Inference: 733.8ms | Downloads: 326 | Likes: 2

---

## About These Recommendations

### Selection Criteria

Models in this guide are selected based on:
- **Popularity**: High download counts and community engagement on HuggingFace Hub
- **Performance**: Fast loading and inference times based on benchmark results
- **Compatibility**: Verified WebGPU support for GPU-accelerated browser execution

### Understanding Benchmark Metrics

**Important:** All performance metrics (load time, inference time, etc.) are measured in a controlled benchmark environment. These metrics are useful for **comparing models against each other**, but they may not reflect the actual performance you'll experience in your specific environment. Factors that affect real-world performance include:
- Hardware specifications (CPU, GPU, memory)
- Browser type and version
- Operating system
- Network conditions (for model loading)
- Concurrent processes and system load

**We recommend** benchmarking models in your own environment with your actual use case to get accurate performance measurements.

### For Production Use

These recommendations are optimized for first-time trials and learning. For production applications, consider:
- Evaluating multiple models for your specific use case
- Testing with your actual data and performance requirements
- Reviewing the full benchmark results for comprehensive comparisons
- Exploring specialized models that may better fit your needs

Visit the full leaderboard to explore all available models and their benchmark results.
