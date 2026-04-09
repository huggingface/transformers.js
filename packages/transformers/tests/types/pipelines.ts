/**
 * The pipeline function should correctly infer:
 *  1. The type of the pipeline, based on the task name.
 *  2. The output type of the pipeline, based on the types of the inputs.
 *
 * To test this, we create pipelines for various tasks, and call them with different types of inputs.
 * We then check that the output types are as expected.
 *
 * Note: These tests are not meant to be executed, but rather to be type-checked by TypeScript.
 */
import { pipeline, InterruptableStoppingCriteria } from "../../src/transformers.js";

import type { RawImage } from "../../src/utils/image.js";
import type { RawAudio } from "../../src/utils/audio.js";
import type { DataArray, Tensor } from "../../src/utils/tensor.js";

import type { BoundingBox } from "../../src/pipelines/_base.js";

import type { AudioClassificationPipeline, AudioClassificationOutput } from "../../src/pipelines/audio-classification.js";
import type { AutomaticSpeechRecognitionPipeline, AutomaticSpeechRecognitionOutput } from "../../src/pipelines/automatic-speech-recognition.js";
import type { BackgroundRemovalPipeline } from "../../src/pipelines/background-removal.js";
import type { DepthEstimationPipeline, DepthEstimationOutput } from "../../src/pipelines/depth-estimation.js";
import type { DocumentQuestionAnsweringPipeline, DocumentQuestionAnsweringOutput } from "../../src/pipelines/document-question-answering.js";
import type { FeatureExtractionPipeline } from "../../src/pipelines/feature-extraction.js";
import type { FillMaskPipeline, FillMaskOutput } from "../../src/pipelines/fill-mask.js";
import type { ImageClassificationPipeline, ImageClassificationOutput } from "../../src/pipelines/image-classification.js";
import type { ImageFeatureExtractionPipeline } from "../../src/pipelines/image-feature-extraction.js";
import type { ImageSegmentationPipeline, ImageSegmentationOutput } from "../../src/pipelines/image-segmentation.js";
import type { ImageToImagePipeline } from "../../src/pipelines/image-to-image.js";
import type { ImageToTextPipeline, ImageToTextOutput } from "../../src/pipelines/image-to-text.js";
import type { ObjectDetectionPipeline, ObjectDetectionOutput } from "../../src/pipelines/object-detection.js";
import type { QuestionAnsweringOutput, QuestionAnsweringPipeline } from "../../src/pipelines/question-answering.js";
import type { SummarizationPipeline, SummarizationOutput } from "../../src/pipelines/summarization.js";
import type { TextClassificationPipeline, TextClassificationOutput } from "../../src/pipelines/text-classification.js";
import type { TextGenerationPipeline, TextGenerationStringOutput, TextGenerationChatOutput, Chat } from "../../src/pipelines/text-generation";
import type { TextToAudioPipeline } from "../../src/pipelines/text-to-audio.js";
import type { Text2TextGenerationPipeline, Text2TextGenerationOutput } from "../../src/pipelines/text2text-generation.js";
import type { TokenClassificationPipeline, TokenClassificationOutput } from "../../src/pipelines/token-classification.js";
import type { TranslationPipeline, TranslationOutput } from "../../src/pipelines/translation.js";
import type { ZeroShotAudioClassificationPipeline, ZeroShotAudioClassificationOutput } from "../../src/pipelines/zero-shot-audio-classification.js";
import type { ZeroShotClassificationPipeline, ZeroShotClassificationOutput } from "../../src/pipelines/zero-shot-classification.js";
import type { ZeroShotImageClassificationPipeline, ZeroShotImageClassificationOutput } from "../../src/pipelines/zero-shot-image-classification.js";
import type { ZeroShotObjectDetectionPipeline, ZeroShotObjectDetectionOutput } from "../../src/pipelines/zero-shot-object-detection.js";

import type { Expect, Equal } from "./_base.ts";

// Dummy inputs
const MODEL_ID = "organization/model";
const URL = "https://example.com";
const TEXT = "This is a test.";
const MESSAGES = [{ role: "user", content: "Hello!" }];
const FLOAT32 = new Float32Array(16000);

// Audio Classification
{
  const classifier = await pipeline("audio-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, AudioClassificationPipeline>>;

  // (a) Single input -> AudioClassificationOutput
  {
    const output = await classifier(URL);
    type T = Expect<Equal<typeof output, AudioClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (b) Batch input -> AudioClassificationOutput[]
  {
    const output = await classifier([URL, URL]);
    type T = Expect<Equal<typeof output, AudioClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
  }
}

// Automatic Speech Recognition
{
  const transcriber = await pipeline("automatic-speech-recognition", MODEL_ID);
  type T = Expect<Equal<typeof transcriber, AutomaticSpeechRecognitionPipeline>>;

  // (a) Single input -> AutomaticSpeechRecognitionOutput
  {
    const output = await transcriber(FLOAT32);
    type T = Expect<Equal<typeof output, AutomaticSpeechRecognitionOutput>>;
    type T1 = Expect<Equal<typeof output["text"], string>>;
  }

  // (b) Batch input -> AutomaticSpeechRecognitionOutput[]
  {
    const output = await transcriber([FLOAT32, FLOAT32]);
    type T = Expect<Equal<typeof output, AutomaticSpeechRecognitionOutput[]>>;
    type T1 = Expect<Equal<typeof output[0]["text"], string>>;
  }
}

// Background Removal
{
  const remover = await pipeline("background-removal", MODEL_ID);
  type T = Expect<Equal<typeof remover, BackgroundRemovalPipeline>>;

  // (a) Single input -> RawImage
  {
    const output = await remover(URL);
    type T = Expect<Equal<typeof output, RawImage>>;
    type T1 = Expect<Equal<typeof output["width"], number>>;
    type T2 = Expect<Equal<typeof output["height"], number>>;
    type T3 = Expect<Equal<typeof output["data"], Uint8ClampedArray | Uint8Array>>;
  }

  // (b) Batch input -> RawImage[]
  {
    const output = await remover([URL, URL]);
    type T = Expect<Equal<typeof output, RawImage[]>>;
    type T1 = Expect<Equal<typeof output[0]["width"], number>>;
    type T2 = Expect<Equal<typeof output[0]["height"], number>>;
    type T3 = Expect<Equal<typeof output[0]["data"], Uint8ClampedArray | Uint8Array>>;
  }
}

// Depth Estimation
{
  const depth_estimator = await pipeline("depth-estimation", MODEL_ID);
  type T = Expect<Equal<typeof depth_estimator, DepthEstimationPipeline>>;

  // (a) Single input -> DepthEstimationOutput
  {
    const output = await depth_estimator(URL);
    type T = Expect<Equal<typeof output, DepthEstimationOutput>>;
    type T1 = Expect<Equal<typeof output["depth"], RawImage>>;
    type T2 = Expect<Equal<typeof output["predicted_depth"], Tensor>>;
  }
  // (b) Batch input with single image -> DepthEstimationOutput[]
  {
    const output = await depth_estimator([URL]);
    type T = Expect<Equal<typeof output, DepthEstimationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0]["depth"], RawImage>>;
    type T2 = Expect<Equal<typeof output[0]["predicted_depth"], Tensor>>;
  }

  // (c) Batch input with multiple images -> DepthEstimationOutput[]
  {
    const output = await depth_estimator([URL, URL]);
    type T = Expect<Equal<typeof output, DepthEstimationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0]["depth"], RawImage>>;
    type T2 = Expect<Equal<typeof output[0]["predicted_depth"], Tensor>>;
  }
}

// Document Question Answering
{
  const answerer = await pipeline("document-question-answering", MODEL_ID);
  type T = Expect<Equal<typeof answerer, DocumentQuestionAnsweringPipeline>>;

  // (a) Single input -> DocumentQuestionAnsweringOutput
  {
    const output = await answerer(URL, TEXT);
    type T = Expect<Equal<typeof output, DocumentQuestionAnsweringOutput>>;
    type T1 = Expect<Equal<typeof output[0]["answer"], string>>;
  }

  // (b) Batch input (=1) -> DocumentQuestionAnsweringOutput
  // TODO: Support batch_size > 1
  {
    const output = await answerer([URL], TEXT);
    type T = Expect<Equal<typeof output, DocumentQuestionAnsweringOutput>>;
    type T1 = Expect<Equal<typeof output[0]["answer"], string>>;
  }
}

// Feature Extraction
{
  const extractor = await pipeline("feature-extraction", MODEL_ID);
  type T = Expect<Equal<typeof extractor, FeatureExtractionPipeline>>;

  // (a) Single input -> Tensor
  {
    const output = await extractor(TEXT);
    type T = Expect<Equal<typeof output, Tensor>>;
    type T1 = Expect<Equal<typeof output["dims"], number[]>>;
    type T2 = Expect<Equal<typeof output["data"], DataArray>>;
  }

  // (b) Batch input -> Tensor
  {
    const output = await extractor([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, Tensor>>;
    type T1 = Expect<Equal<typeof output["dims"], number[]>>;
    type T2 = Expect<Equal<typeof output["data"], DataArray>>;
  }
}

// Fill-Mask
{
  const unmasker = await pipeline("fill-mask", MODEL_ID);
  type T = Expect<Equal<typeof unmasker, FillMaskPipeline>>;

  // (a) Single input -> FillMaskOutput
  {
    const output = await unmasker("This is a <mask> test.");
    type T = Expect<Equal<typeof output, FillMaskOutput>>;
    type T1 = Expect<Equal<typeof output[0]["sequence"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[0]["token"], number>>;
    type T4 = Expect<Equal<typeof output[0]["token_str"], string>>;
  }

  // (b) Batch input -> FillMaskOutput[]
  {
    const output = await unmasker(["This is a <mask> test.", "Another <mask> example."]);
    type T = Expect<Equal<typeof output, FillMaskOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["sequence"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[0][0]["token"], number>>;
    type T4 = Expect<Equal<typeof output[0][0]["token_str"], string>>;
  }
}

// Image Classification
{
  const classifier = await pipeline("image-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, ImageClassificationPipeline>>;

  // (a) Single input -> ImageClassificationOutput
  {
    const output = await classifier(URL);
    type T = Expect<Equal<typeof output, ImageClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (b) Batch input -> ImageClassificationOutput[]
  {
    const output = await classifier([URL, URL]);
    type T = Expect<Equal<typeof output, ImageClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
  }
}

// Image Feature Extraction
{
  const image_extractor = await pipeline("image-feature-extraction", MODEL_ID);
  type T = Expect<Equal<typeof image_extractor, ImageFeatureExtractionPipeline>>;

  // (a) Single input -> Tensor
  {
    const output = await image_extractor(URL);
    type T = Expect<Equal<typeof output, Tensor>>;
    type T1 = Expect<Equal<typeof output["dims"], number[]>>;
    type T2 = Expect<Equal<typeof output["data"], DataArray>>;
  }

  // (b) Batch input -> Tensor
  {
    const output = await image_extractor([URL, URL]);
    type T = Expect<Equal<typeof output, Tensor>>;
    type T1 = Expect<Equal<typeof output["dims"], number[]>>;
    type T2 = Expect<Equal<typeof output["data"], DataArray>>;
  }
}

// Image Segmentation
{
  const segmenter = await pipeline("image-segmentation", MODEL_ID);
  type T = Expect<Equal<typeof segmenter, ImageSegmentationPipeline>>;

  // (a) Single input -> ImageSegmentationOutput
  {
    const output = await segmenter(URL);
    type T = Expect<Equal<typeof output, ImageSegmentationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string | null>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number | null>>;
    type T3 = Expect<Equal<typeof output[0]["mask"], RawImage>>;
  }

  // (b) Batch input (=1) -> ImageSegmentationOutput[]
  // TODO: Support batch_size > 1
  {
    const output = await segmenter([URL]);
    type T = Expect<Equal<typeof output, ImageSegmentationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string | null>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number | null>>;
    type T3 = Expect<Equal<typeof output[0]["mask"], RawImage>>;
  }
}

// Image-to-Image
{
  const upscaler = await pipeline("image-to-image", MODEL_ID);
  type T = Expect<Equal<typeof upscaler, ImageToImagePipeline>>;

  // (a) Single input -> RawImage
  {
    const output = await upscaler(URL);
    type T = Expect<Equal<typeof output, RawImage>>;
    type T1 = Expect<Equal<typeof output["width"], number>>;
    type T2 = Expect<Equal<typeof output["height"], number>>;
    type T3 = Expect<Equal<typeof output["data"], Uint8ClampedArray | Uint8Array>>;
  }

  // (b) Batch input -> RawImage[]
  {
    const output = await upscaler([URL, URL]);
    type T = Expect<Equal<typeof output, RawImage[]>>;
    type T1 = Expect<Equal<typeof output[0]["width"], number>>;
    type T2 = Expect<Equal<typeof output[0]["height"], number>>;
    type T3 = Expect<Equal<typeof output[0]["data"], Uint8ClampedArray | Uint8Array>>;
  }
}

// Image-to-Text
{
  const ocr = await pipeline("image-to-text", MODEL_ID);
  type T = Expect<Equal<typeof ocr, ImageToTextPipeline>>;

  // (a) Single input -> ImageToTextOutput
  {
    const output = await ocr(URL);
    type T = Expect<Equal<typeof output, ImageToTextOutput>>;
    type T1 = Expect<Equal<typeof output[0]["generated_text"], string>>;
  }

  // (b) Batch input -> ImageToTextOutput[]
  {
    const output = await ocr([URL, URL]);
    type T = Expect<Equal<typeof output, ImageToTextOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["generated_text"], string>>;
  }
}

// Object Detection
{
  const detector = await pipeline("object-detection", MODEL_ID);
  type T = Expect<Equal<typeof detector, ObjectDetectionPipeline>>;

  // (a) Single input -> ObjectDetectionOutput
  {
    const output = await detector(URL);
    type T = Expect<Equal<typeof output, ObjectDetectionOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[0]["box"], BoundingBox>>;
  }

  // (b) Batch input -> ObjectDetectionOutput[]
  {
    const output = await detector([URL, URL]);
    type T = Expect<Equal<typeof output, ObjectDetectionOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[0][0]["box"], BoundingBox>>;
  }
}

// Question Answering
{
  const answerer = await pipeline("question-answering", MODEL_ID);
  type T = Expect<Equal<typeof answerer, QuestionAnsweringPipeline>>;

  // (a) Single input, top_k=1 -> QuestionAnsweringOutput
  {
    const output = await answerer(TEXT, TEXT, { top_k: 1 });
    type T = Expect<Equal<typeof output, QuestionAnsweringOutput>>;
    type T1 = Expect<Equal<typeof output["answer"], string>>;
    type T2 = Expect<Equal<typeof output["score"], number>>;
  }

  // (b) Single input, top_k=3 -> QuestionAnsweringOutput[]
  {
    const output = await answerer(TEXT, TEXT, { top_k: 3 });
    type T = Expect<Equal<typeof output, QuestionAnsweringOutput[]>>;
    type T1 = Expect<Equal<typeof output[0]["answer"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (c) Batch input, top_k=1 -> QuestionAnsweringOutput[]
  {
    const output = await answerer([TEXT, TEXT], [TEXT, TEXT]);
    type T = Expect<Equal<typeof output, QuestionAnsweringOutput[]>>;
    type T1 = Expect<Equal<typeof output[0]["answer"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (d) Batch input, top_k=2 -> QuestionAnsweringOutput[][]
  {
    const output = await answerer([TEXT, TEXT], [TEXT, TEXT], { top_k: 2 });
    type T = Expect<Equal<typeof output, QuestionAnsweringOutput[][]>>;
    type T1 = Expect<Equal<typeof output[0][0]["answer"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
  }
}

// Summarization
{
  const summarizer = await pipeline("summarization", MODEL_ID);
  type T = Expect<Equal<typeof summarizer, SummarizationPipeline>>;

  // (a) Single input -> SummarizationOutput
  {
    const output = await summarizer(TEXT);
    type T = Expect<Equal<typeof output, SummarizationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["summary_text"], string>>;
  }

  // (b) Batch input -> SummarizationOutput
  {
    const output = await summarizer([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, SummarizationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["summary_text"], string>>;
  }
}

// Text Classification
{
  // Create a text classification pipeline
  const classifier = await pipeline("text-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, TextClassificationPipeline>>;

  // (a) Single input, top_k=1 -> TextClassificationOutput
  {
    const output = await classifier(TEXT, { top_k: 1 });
    type T = Expect<Equal<typeof output, TextClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (b) Single input, top_k=2 -> TextClassificationOutput
  {
    const output = await classifier(TEXT, { top_k: 2 });
    type T = Expect<Equal<typeof output, TextClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[1]["label"], string>>;
    type T4 = Expect<Equal<typeof output[1]["score"], number>>;
  }

  // (c) Batch input, top_k=1 -> TextClassificationOutput
  {
    const output = await classifier([TEXT, TEXT], { top_k: 1 });
    type T = Expect<Equal<typeof output, TextClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (d) Batch input, top_k=2 -> TextClassificationOutput[]
  {
    const output = await classifier([TEXT, TEXT], { top_k: 2 });
    type T = Expect<Equal<typeof output, TextClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[1][0]["label"], string>>;
    type T4 = Expect<Equal<typeof output[1][0]["score"], number>>;
  }
}

// Text Generation
{
  const generator = await pipeline("text-generation", MODEL_ID);
  type T = Expect<Equal<typeof generator, TextGenerationPipeline>>;

  // (a) Single input -> TextGenerationStringOutput
  {
    const output = await generator(TEXT);
    type T = Expect<Equal<typeof output, TextGenerationStringOutput>>;
    type T1 = Expect<Equal<typeof output[0]["generated_text"], string>>;
  }

  // (b) Batch input -> TextGenerationStringOutput[]
  {
    const output = await generator([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, TextGenerationStringOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["generated_text"], string>>;
  }

  // (c) Chat input -> TextGenerationChatOutput
  {
    const output = await generator(MESSAGES);
    type T = Expect<Equal<typeof output, TextGenerationChatOutput>>;
    type T1 = Expect<Equal<typeof output[0]["generated_text"], Chat>>;
  }

  // (d) Batch chat input -> TextGenerationChatOutput[]
  {
    const output = await generator([MESSAGES, MESSAGES]);
    type T = Expect<Equal<typeof output, TextGenerationChatOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["generated_text"], Chat>>;
  }

  // (e) Chat input with generation parameters -> TextGenerationChatOutput
  {
    const output = await generator(MESSAGES, {
      max_new_tokens: 50,
      stopping_criteria: [new InterruptableStoppingCriteria()],
    });
    type T = Expect<Equal<typeof output, TextGenerationChatOutput>>;
    type T1 = Expect<Equal<typeof output[0]["generated_text"], Chat>>;
  }
}

// Text-to-Audio
{
  const generator = await pipeline("text-to-audio", MODEL_ID);
  type T = Expect<Equal<typeof generator, TextToAudioPipeline>>;

  // (a) Single input -> RawAudio
  {
    const output = await generator(TEXT);
    type T = Expect<Equal<typeof output, RawAudio>>;
    type T1 = Expect<Equal<typeof output["data"], Float32Array>>;
    type T2 = Expect<Equal<typeof output["sampling_rate"], number>>;
  }

  // (b) Batch input -> RawAudio[]
  {
    const output = await generator([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, RawAudio[]>>;
    type T1 = Expect<Equal<typeof output[0]["data"], Float32Array>>;
    type T2 = Expect<Equal<typeof output[0]["sampling_rate"], number>>;
  }
}

// Text2Text Generation
{
  const generator = await pipeline("text2text-generation", MODEL_ID);
  type T = Expect<Equal<typeof generator, Text2TextGenerationPipeline>>;

  // (a) Single input -> Text2TextGenerationOutput
  {
    const output = await generator(TEXT);
    type T = Expect<Equal<typeof output, Text2TextGenerationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["generated_text"], string>>;
  }

  // (b) Batch input -> Text2TextGenerationOutput
  {
    const output = await generator([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, Text2TextGenerationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["generated_text"], string>>;
  }
}

// Token Classification
{
  const classifier = await pipeline("token-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, TokenClassificationPipeline>>;

  // (a) Single input -> TokenClassificationOutput
  {
    const output = await classifier(TEXT);
    type T = Expect<Equal<typeof output, TokenClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["word"], string>>;
    type T2 = Expect<Equal<typeof output[0]["entity"], string>>;
    type T3 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (b) Batch input -> TokenClassificationOutput[]
  {
    const output = await classifier([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, TokenClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["word"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["entity"], string>>;
    type T3 = Expect<Equal<typeof output[0][0]["score"], number>>;
  }
}

// Translation
{
  const translator = await pipeline("translation", MODEL_ID);
  type T = Expect<Equal<typeof translator, TranslationPipeline>>;

  // (a) Single input -> TranslationOutput
  {
    const output = await translator(TEXT);
    type T = Expect<Equal<typeof output, TranslationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["translation_text"], string>>;
  }

  // (b) Batch input -> TranslationOutput
  {
    const output = await translator([TEXT, TEXT]);
    type T = Expect<Equal<typeof output, TranslationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["translation_text"], string>>;
  }
}

// Zero-shot Audio Classification
{
  const classifier = await pipeline("zero-shot-audio-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, ZeroShotAudioClassificationPipeline>>;

  // (a) Single input -> ZeroShotAudioClassificationOutput
  {
    const output = await classifier(FLOAT32, ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotAudioClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (b) Batch input -> ZeroShotAudioClassificationOutput[]
  {
    const output = await classifier([FLOAT32, FLOAT32], ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotAudioClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
  }
}

// Zero-shot Classification
{
  const classifier = await pipeline("zero-shot-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, ZeroShotClassificationPipeline>>;

  // (a) Single input -> ZeroShotClassificationOutput
  {
    const output = await classifier(TEXT, ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotClassificationOutput>>;
    type T1 = Expect<Equal<typeof output["sequence"], string>>;
    type T2 = Expect<Equal<typeof output["labels"], string[]>>;
    type T3 = Expect<Equal<typeof output["scores"], number[]>>;
  }

  // (b) Batch input -> ZeroShotClassificationOutput[]
  {
    const output = await classifier([TEXT, TEXT], ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0]["sequence"], string>>;
    type T2 = Expect<Equal<typeof output[0]["labels"], string[]>>;
    type T3 = Expect<Equal<typeof output[0]["scores"], number[]>>;
  }
}

// Zero-shot Image Classification
{
  const classifier = await pipeline("zero-shot-image-classification", MODEL_ID);
  type T = Expect<Equal<typeof classifier, ZeroShotImageClassificationPipeline>>;

  // (a) Single input -> ZeroShotImageClassificationOutput
  {
    const output = await classifier(URL, ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotImageClassificationOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
  }

  // (b) Batch input -> ZeroShotImageClassificationOutput[]
  {
    const output = await classifier([URL, URL], ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotImageClassificationOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
  }
}

// Zero-shot Object Detection
{
  const detector = await pipeline("zero-shot-object-detection", MODEL_ID);
  type T = Expect<Equal<typeof detector, ZeroShotObjectDetectionPipeline>>;

  // (a) Single input -> ZeroShotObjectDetectionOutput
  {
    const output = await detector(URL, ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotObjectDetectionOutput>>;
    type T1 = Expect<Equal<typeof output[0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[0]["box"], BoundingBox>>;
  }

  // (b) Batch input -> ZeroShotObjectDetectionOutput[]
  {
    const output = await detector([URL, URL], ["class A", "class B"]);
    type T = Expect<Equal<typeof output, ZeroShotObjectDetectionOutput[]>>;
    type T1 = Expect<Equal<typeof output[0][0]["label"], string>>;
    type T2 = Expect<Equal<typeof output[0][0]["score"], number>>;
    type T3 = Expect<Equal<typeof output[0][0]["box"], BoundingBox>>;
  }
}
