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
import { pipeline } from "../../src/transformers.js";

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
import type { TextGenerationPipeline, TextGenerationOutput, TextGenerationChatOutput, Chat } from "../../src/pipelines/text-generation";
import type { TextToAudioPipeline } from "../../src/pipelines/text-to-audio.js";
import type { Text2TextGenerationPipeline, Text2TextGenerationOutput } from "../../src/pipelines/text2text-generation.js";
import type { TokenClassificationPipeline, TokenClassificationOutput } from "../../src/pipelines/token-classification.js";
import type { TranslationPipeline, TranslationOutput } from "../../src/pipelines/translation.js";
import type { ZeroShotAudioClassificationPipeline, ZeroShotAudioClassificationOutput } from "../../src/pipelines/zero-shot-audio-classification.js";
import type { ZeroShotClassificationPipeline, ZeroShotClassificationOutput } from "../../src/pipelines/zero-shot-classification.js";
import type { ZeroShotImageClassificationPipeline, ZeroShotImageClassificationOutput } from "../../src/pipelines/zero-shot-image-classification.js";
import type { ZeroShotObjectDetectionPipeline, ZeroShotObjectDetectionOutput } from "../../src/pipelines/zero-shot-object-detection.js";

// Dummy inputs
const MODEL_ID = "organization/model";
const URL = "https://example.com";
const TEXT = "This is a test.";
const MESSAGES = [{ role: "user", content: "Hello!" }];
const FLOAT32 = new Float32Array(16000);

// Audio Classification
{
  const classifier = await pipeline("audio-classification", MODEL_ID);
  classifier as AudioClassificationPipeline;

  // (a) Single input -> AudioClassificationOutput
  {
    const output = await classifier(URL);
    output as AudioClassificationOutput;
    output[0].label as string;
    output[0].score as number;
  }

  // (b) Batch input -> AudioClassificationOutput[]
  {
    const output = await classifier([URL, URL]);
    output as AudioClassificationOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
  }
}

// Automatic Speech Recognition
{
  const transcriber = await pipeline("automatic-speech-recognition", MODEL_ID);
  transcriber as AutomaticSpeechRecognitionPipeline;

  // (a) Single input -> AutomaticSpeechRecognitionOutput
  {
    const output = await transcriber(FLOAT32);
    output as AutomaticSpeechRecognitionOutput;
    output.text as string;
  }

  // (b) Batch input -> AutomaticSpeechRecognitionOutput[]
  {
    const output = await transcriber([FLOAT32, FLOAT32]);
    output as AutomaticSpeechRecognitionOutput[];
    output[0].text as string;
  }
}

// Background Removal
{
  const remover = await pipeline("background-removal", MODEL_ID);
  remover as BackgroundRemovalPipeline;

  // (a) Single input -> RawImage
  {
    const output = await remover(URL);
    output as RawImage;
    output.width as number;
    output.height as number;
    output.data as Uint8ClampedArray;
  }

  // (b) Batch input -> RawImage[]
  {
    const output = await remover([URL, URL]);
    output as RawImage[];
    output[0].width as number;
    output[0].height as number;
    output[0].data as Uint8ClampedArray;
  }
}

// Depth Estimation
{
  const depth_estimator = await pipeline("depth-estimation", MODEL_ID);
  depth_estimator as DepthEstimationPipeline;

  // (a) Single input -> DepthEstimationOutput
  {
    const output = await depth_estimator(URL);
    output as DepthEstimationOutput;
    output.depth as RawImage;
    output.predicted_depth as Tensor;
  }
  // (b) Batch input with single image -> DepthEstimationOutput[]
  {
    const output = await depth_estimator([URL]);
    output as DepthEstimationOutput[];
    output[0].depth as RawImage;
    output[0].predicted_depth as Tensor;
  }

  // (c) Batch input with multiple images -> DepthEstimationOutput[]
  {
    const output = await depth_estimator([URL, URL]);
    output as DepthEstimationOutput[];
    output[0].depth as RawImage;
    output[0].predicted_depth as Tensor;
  }
}

// Document Question Answering
{
  const answerer = await pipeline("document-question-answering", MODEL_ID);
  answerer as DocumentQuestionAnsweringPipeline;

  // (a) Single input -> DocumentQuestionAnsweringOutput
  {
    const output = await answerer(URL, TEXT);
    output as DocumentQuestionAnsweringOutput;
    output[0].answer as string;
  }

  // (b) Batch input (=1) -> DocumentQuestionAnsweringOutput
  // TODO: Support batch_size > 1
  {
    const output = await answerer([URL], TEXT);
    output as DocumentQuestionAnsweringOutput;
    output[0].answer as string;
  }
}

// Feature Extraction
{
  const extractor = await pipeline("feature-extraction", MODEL_ID);
  extractor as FeatureExtractionPipeline;

  // (a) Single input -> Tensor
  {
    const output = await extractor(TEXT);
    output as Tensor;
    output.dims as number[];
    output.data as DataArray;
  }

  // (b) Batch input -> Tensor
  {
    const output = await extractor([TEXT, TEXT]);
    output as Tensor;
    output.dims as number[];
    output.data as DataArray;
  }
}

// Fill-Mask
{
  const unmasker = await pipeline("fill-mask", MODEL_ID);
  unmasker as FillMaskPipeline;

  // (a) Single input -> FillMaskOutput
  {
    const output = await unmasker("This is a <mask> test.");
    output as FillMaskOutput;
    output[0].sequence as string;
    output[0].score as number;
    output[0].token as number;
    output[0].token_str as string;
  }

  // (b) Batch input -> FillMaskOutput[]
  {
    const output = await unmasker(["This is a <mask> test.", "Another <mask> example."]);
    output as FillMaskOutput[];
    output[0][0].sequence as string;
    output[0][0].score as number;
    output[0][0].token as number;
    output[0][0].token_str as string;
  }
}

// Image Classification
{
  const classifier = await pipeline("image-classification", MODEL_ID);
  classifier as ImageClassificationPipeline;

  // (a) Single input -> ImageClassificationOutput
  {
    const output = await classifier(URL);
    output as ImageClassificationOutput;
    output[0].label as string;
    output[0].score as number;
  }

  // (b) Batch input -> ImageClassificationOutput[]
  {
    const output = await classifier([URL, URL]);
    output as ImageClassificationOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
  }
}

// Image Feature Extraction
{
  const image_extractor = await pipeline("image-feature-extraction", MODEL_ID);
  image_extractor as ImageFeatureExtractionPipeline;

  // (a) Single input -> Tensor
  {
    const output = await image_extractor(URL);
    output as Tensor;
    output.dims as number[];
    output.data as DataArray;
  }

  // (b) Batch input -> Tensor
  {
    const output = await image_extractor([URL, URL]);
    output as Tensor;
    output.dims as number[];
    output.data as DataArray;
  }
}

// Image Segmentation
{
  const segmenter = await pipeline("image-segmentation", MODEL_ID);
  segmenter as ImageSegmentationPipeline;

  // (a) Single input -> ImageSegmentationOutput
  {
    const output = await segmenter(URL);
    output as ImageSegmentationOutput;
    output[0].label as string;
    output[0].score as number;
    output[0].mask as RawImage;
  }

  // (b) Batch input (=1) -> ImageSegmentationOutput[]
  // TODO: Support batch_size > 1
  {
    const output = await segmenter([URL]);
    output as ImageSegmentationOutput;
    output[0].label as string;
    output[0].score as number;
    output[0].mask as RawImage;
  }
}

// Image-to-Image
{
  const upscaler = await pipeline("image-to-image", MODEL_ID);
  upscaler as ImageToImagePipeline;

  // (a) Single input -> RawImage
  {
    const output = await upscaler(URL);
    output as RawImage;
    output.width as number;
    output.height as number;
    output.data as Uint8ClampedArray;
  }

  // (b) Batch input -> RawImage[]
  {
    const output = await upscaler([URL, URL]);
    output as RawImage[];
    output[0].width as number;
    output[0].height as number;
    output[0].data as Uint8ClampedArray;
  }
}

// Image-to-Text
{
  const ocr = await pipeline("image-to-text", MODEL_ID);
  ocr as ImageToTextPipeline;

  // (a) Single input -> ImageToTextOutput
  {
    const output = await ocr(URL);
    output as ImageToTextOutput;
    output[0].generated_text as string;
  }

  // (b) Batch input -> ImageToTextOutput[]
  {
    const output = await ocr([URL, URL]);
    output as ImageToTextOutput[];
    output[0][0].generated_text as string;
  }
}

// Object Detection
{
  const detector = await pipeline("object-detection", MODEL_ID);
  detector as ObjectDetectionPipeline;

  // (a) Single input -> ObjectDetectionOutput
  {
    const output = await detector(URL);
    output as ObjectDetectionOutput;
    output[0].label as string;
    output[0].score as number;
    output[0].box as BoundingBox;
  }

  // (b) Batch input -> ObjectDetectionOutput[]
  {
    const output = await detector([URL, URL]);
    output as ObjectDetectionOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
    output[0][0].box as BoundingBox;
  }
}

// Question Answering
{
  const answerer = await pipeline("question-answering", MODEL_ID);
  answerer as QuestionAnsweringPipeline;

  // (a) Single input, top_k=1 -> QuestionAnsweringOutput
  {
    const output = await answerer(TEXT, TEXT, { top_k: 1 });
    output as QuestionAnsweringOutput;
    output.answer as string;
    output.score as number;
  }

  // (b) Single input, top_k=3 -> QuestionAnsweringOutput[]
  {
    const output = await answerer(TEXT, TEXT, { top_k: 3 });
    output as QuestionAnsweringOutput[];
    output[0].answer as string;
    output[0].score as number;
  }

  // (c) Batch input, top_k=1 -> QuestionAnsweringOutput[]
  {
    const output = await answerer([TEXT, TEXT], [TEXT, TEXT]);
    output as QuestionAnsweringOutput[];
    output[0].answer as string;
    output[0].score as number;
  }

  // (d) Batch input, top_k=2 -> QuestionAnsweringOutput[][]
  {
    const output = await answerer([TEXT, TEXT], [TEXT, TEXT], { top_k: 2 });
    output as QuestionAnsweringOutput[][];
    output[0][0].answer as string;
    output[0][0].score as number;
  }
}

// Summarization
{
  const summarizer = await pipeline("summarization", MODEL_ID);
  summarizer as SummarizationPipeline;

  // (a) Single input -> SummarizationOutput
  {
    const output = await summarizer(TEXT);
    output as SummarizationOutput;
    output[0].summary_text as string;
  }

  // (b) Batch input -> SummarizationOutput
  {
    const output = await summarizer([TEXT, TEXT]);
    output as SummarizationOutput;
    output[0].summary_text as string;
  }
}

// Text Classification
{
  // Create a text classification pipeline
  const classifier = await pipeline("text-classification", MODEL_ID);
  classifier as TextClassificationPipeline;

  // (a) Single input, top_k=1 -> TextClassificationOutput
  {
    const output = await classifier(TEXT, { top_k: 1 });
    output as TextClassificationOutput;
    output[0].label as string;
    output[0].score as number;
  }

  // (b) Single input, top_k=2 -> TextClassificationOutput
  {
    const output = await classifier(TEXT, { top_k: 2 });
    output as TextClassificationOutput;
    output[0].label as string;
    output[0].score as number;
    output[1].label as string;
    output[1].score as number;
  }

  // (c) Batch input, top_k=1 -> TextClassificationOutput
  {
    const output = await classifier([TEXT, TEXT], { top_k: 1 });
    output as TextClassificationOutput;
    output[0].label as string;
    output[0].score as number;
  }

  // (d) Batch input, top_k=2 -> TextClassificationOutput[]
  {
    const output = await classifier([TEXT, TEXT], { top_k: 2 });
    output as TextClassificationOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
    output[1][0].label as string;
    output[1][0].score as number;
  }
}

// Text Generation
{
  const generator = await pipeline("text-generation", MODEL_ID);
  generator as TextGenerationPipeline;

  // (a) Single input -> TextGenerationOutput
  {
    const output = await generator(TEXT);
    output as TextGenerationOutput;
    output[0].generated_text as string;
  }

  // (b) Batch input -> TextGenerationOutput[]
  {
    const output = await generator([TEXT, TEXT]);
    output as TextGenerationOutput[];
    output[0][0].generated_text as string;
  }

  // (c) Chat input -> TextGenerationChatOutput
  {
    const output = await generator(MESSAGES);
    output as TextGenerationChatOutput;
    output[0].generated_text as Chat;
    output[0].generated_text.at(-1)!.role as string;
    output[0].generated_text.at(-1)!.content as string;
  }

  // (d) Batch chat input -> TextGenerationChatOutput[]
  {
    const output = await generator([MESSAGES, MESSAGES]);
    output as TextGenerationChatOutput[];
    output[0][0].generated_text as Chat;
    output[0][0].generated_text.at(-1)!.role as string;
    output[0][0].generated_text.at(-1)!.content as string;
  }
}

// Text-to-Audio
{
  const generator = await pipeline("text-to-audio", MODEL_ID);
  generator as TextToAudioPipeline;

  // (a) Single input -> RawAudio
  {
    const output = await generator(TEXT);
    output as RawAudio;
    output.data as Float32Array;
    output.sampling_rate as number;
  }

  // (b) Batch input -> RawAudio[]
  {
    const output = await generator([TEXT, TEXT]);
    output as RawAudio[];
    output[0].data as Float32Array;
    output[0].sampling_rate as number;
  }
}

// Text2Text Generation
{
  const generator = await pipeline("text2text-generation", MODEL_ID);
  generator as Text2TextGenerationPipeline;

  // (a) Single input -> Text2TextGenerationOutput
  {
    const output = await generator(TEXT);
    output as Text2TextGenerationOutput;
    output[0].generated_text as string;
  }

  // (b) Batch input -> Text2TextGenerationOutput
  {
    const output = await generator([TEXT, TEXT]);
    output as Text2TextGenerationOutput;
    output[0].generated_text as string;
  }
}

// Token Classification
{
  const classifier = await pipeline("token-classification", MODEL_ID);
  classifier as TokenClassificationPipeline;

  // (a) Single input -> TokenClassificationOutput
  {
    const output = await classifier(TEXT);
    output as TokenClassificationOutput;
    output[0].word as string;
    output[0].entity as string;
    output[0].score as number;
  }

  // (b) Batch input -> TokenClassificationOutput[]
  {
    const output = await classifier([TEXT, TEXT]);
    output as TokenClassificationOutput[];
    output[0][0].word as string;
    output[0][0].entity as string;
    output[0][0].score as number;
  }
}

// Translation
{
  const translator = await pipeline("translation", MODEL_ID);
  translator as TranslationPipeline;

  // (a) Single input -> TranslationOutput
  {
    const output = await translator(TEXT);
    output as TranslationOutput;
    output[0].translation_text as string;
  }

  // (b) Batch input -> TranslationOutput
  {
    const output = await translator([TEXT, TEXT]);
    output as TranslationOutput;
    output[0].translation_text as string;
  }
}

// Zero-shot Audio Classification
{
  const classifier = await pipeline("zero-shot-audio-classification", MODEL_ID);
  classifier as ZeroShotAudioClassificationPipeline;

  // (a) Single input -> ZeroShotAudioClassificationOutput
  {
    const output = await classifier(FLOAT32, ["class A", "class B"]);
    output as ZeroShotAudioClassificationOutput;
    output[0].label as string;
    output[0].score as number;
  }

  // (b) Batch input -> ZeroShotAudioClassificationOutput[]
  {
    const output = await classifier([FLOAT32, FLOAT32], ["class A", "class B"]);
    output as ZeroShotAudioClassificationOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
  }
}

// Zero-shot Classification
{
  const classifier = await pipeline("zero-shot-classification", MODEL_ID);
  classifier as ZeroShotClassificationPipeline;

  // (a) Single input -> ZeroShotClassificationOutput
  {
    const output = await classifier(TEXT, ["class A", "class B"]);
    output as ZeroShotClassificationOutput;
    output.sequence as string;
    output.labels as string[];
    output.scores as number[];
  }

  // (b) Batch input -> ZeroShotClassificationOutput[]
  {
    const output = await classifier([TEXT, TEXT], ["class A", "class B"]);
    output as ZeroShotClassificationOutput[];
    output[0].sequence as string;
    output[0].labels as string[];
    output[0].scores as number[];
  }
}

// Zero-shot Image Classification
{
  const classifier = await pipeline("zero-shot-image-classification", MODEL_ID);
  classifier as ZeroShotImageClassificationPipeline;

  // (a) Single input -> ZeroShotImageClassificationOutput
  {
    const output = await classifier(URL, ["class A", "class B"]);
    output as ZeroShotImageClassificationOutput;
    output[0].label as string;
    output[0].score as number;
  }

  // (b) Batch input -> ZeroShotImageClassificationOutput[]
  {
    const output = await classifier([URL, URL], ["class A", "class B"]);
    output as ZeroShotImageClassificationOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
  }
}

// Zero-shot Object Detection
{
  const detector = await pipeline("zero-shot-object-detection", MODEL_ID);
  detector as ZeroShotObjectDetectionPipeline;

  // (a) Single input -> ZeroShotObjectDetectionOutput
  {
    const output = await detector(URL, ["class A", "class B"]);
    output as ZeroShotObjectDetectionOutput;
    output[0].label as string;
    output[0].score as number;
    output[0].box as BoundingBox;
  }

  // (b) Batch input -> ZeroShotObjectDetectionOutput[]
  {
    const output = await detector([URL, URL], ["class A", "class B"]);
    output as ZeroShotObjectDetectionOutput[];
    output[0][0].label as string;
    output[0][0].score as number;
    output[0][0].box as BoundingBox;
  }
}
