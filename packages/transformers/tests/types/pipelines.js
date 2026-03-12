"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
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
var transformers_js_1 = require("../../src/transformers.js");
// Dummy inputs
var MODEL_ID = "organization/model";
var URL = "https://example.com";
var TEXT = "This is a test.";
var MESSAGES = [{ role: "user", content: "Hello!" }];
var FLOAT32 = new Float32Array(16000);
// Audio Classification
{
    var classifier = await (0, transformers_js_1.pipeline)("audio-classification", MODEL_ID);
    // (a) Single input -> AudioClassificationOutput
    {
        var output = await classifier(URL);
    }
    // (b) Batch input -> AudioClassificationOutput[]
    {
        var output = await classifier([URL, URL]);
    }
}
// Automatic Speech Recognition
{
    var transcriber = await (0, transformers_js_1.pipeline)("automatic-speech-recognition", MODEL_ID);
    // (a) Single input -> AutomaticSpeechRecognitionOutput
    {
        var output = await transcriber(FLOAT32);
    }
    // (b) Batch input -> AutomaticSpeechRecognitionOutput[]
    {
        var output = await transcriber([FLOAT32, FLOAT32]);
    }
}
// Background Removal
{
    var remover = await (0, transformers_js_1.pipeline)("background-removal", MODEL_ID);
    // (a) Single input -> RawImage
    {
        var output = await remover(URL);
    }
    // (b) Batch input -> RawImage[]
    {
        var output = await remover([URL, URL]);
    }
}
// Depth Estimation
{
    var depth_estimator = await (0, transformers_js_1.pipeline)("depth-estimation", MODEL_ID);
    // (a) Single input -> DepthEstimationOutput
    {
        var output = await depth_estimator(URL);
    }
    // (b) Batch input with single image -> DepthEstimationOutput[]
    {
        var output = await depth_estimator([URL]);
    }
    // (c) Batch input with multiple images -> DepthEstimationOutput[]
    {
        var output = await depth_estimator([URL, URL]);
    }
}
// Document Question Answering
{
    var answerer = await (0, transformers_js_1.pipeline)("document-question-answering", MODEL_ID);
    // (a) Single input -> DocumentQuestionAnsweringOutput
    {
        var output = await answerer(URL, TEXT);
    }
    // (b) Batch input (=1) -> DocumentQuestionAnsweringOutput
    // TODO: Support batch_size > 1
    {
        var output = await answerer([URL], TEXT);
    }
}
// Feature Extraction
{
    var extractor = await (0, transformers_js_1.pipeline)("feature-extraction", MODEL_ID);
    // (a) Single input -> Tensor
    {
        var output = await extractor(TEXT);
    }
    // (b) Batch input -> Tensor
    {
        var output = await extractor([TEXT, TEXT]);
    }
}
// Fill-Mask
{
    var unmasker = await (0, transformers_js_1.pipeline)("fill-mask", MODEL_ID);
    // (a) Single input -> FillMaskOutput
    {
        var output = await unmasker("This is a <mask> test.");
    }
    // (b) Batch input -> FillMaskOutput[]
    {
        var output = await unmasker(["This is a <mask> test.", "Another <mask> example."]);
    }
}
// Image Classification
{
    var classifier = await (0, transformers_js_1.pipeline)("image-classification", MODEL_ID);
    // (a) Single input -> ImageClassificationOutput
    {
        var output = await classifier(URL);
    }
    // (b) Batch input -> ImageClassificationOutput[]
    {
        var output = await classifier([URL, URL]);
    }
}
// Image Feature Extraction
{
    var image_extractor = await (0, transformers_js_1.pipeline)("image-feature-extraction", MODEL_ID);
    // (a) Single input -> Tensor
    {
        var output = await image_extractor(URL);
    }
    // (b) Batch input -> Tensor
    {
        var output = await image_extractor([URL, URL]);
    }
}
// Image Segmentation
{
    var segmenter = await (0, transformers_js_1.pipeline)("image-segmentation", MODEL_ID);
    // (a) Single input -> ImageSegmentationOutput
    {
        var output = await segmenter(URL);
    }
    // (b) Batch input (=1) -> ImageSegmentationOutput[]
    // TODO: Support batch_size > 1
    {
        var output = await segmenter([URL]);
    }
}
// Image-to-Image
{
    var upscaler = await (0, transformers_js_1.pipeline)("image-to-image", MODEL_ID);
    // (a) Single input -> RawImage
    {
        var output = await upscaler(URL);
    }
    // (b) Batch input -> RawImage[]
    {
        var output = await upscaler([URL, URL]);
    }
}
// Image-to-Text
{
    var ocr = await (0, transformers_js_1.pipeline)("image-to-text", MODEL_ID);
    // (a) Single input -> ImageToTextOutput
    {
        var output = await ocr(URL);
    }
    // (b) Batch input -> ImageToTextOutput[]
    {
        var output = await ocr([URL, URL]);
    }
}
// Object Detection
{
    var detector = await (0, transformers_js_1.pipeline)("object-detection", MODEL_ID);
    // (a) Single input -> ObjectDetectionOutput
    {
        var output = await detector(URL);
    }
    // (b) Batch input -> ObjectDetectionOutput[]
    {
        var output = await detector([URL, URL]);
    }
}
// Question Answering
{
    var answerer = await (0, transformers_js_1.pipeline)("question-answering", MODEL_ID);
    // (a) Single input, top_k=1 -> QuestionAnsweringOutput
    {
        var output = await answerer(TEXT, TEXT, { top_k: 1 });
    }
    // (b) Single input, top_k=3 -> QuestionAnsweringOutput[]
    {
        var output = await answerer(TEXT, TEXT, { top_k: 3 });
    }
    // (c) Batch input, top_k=1 -> QuestionAnsweringOutput[]
    {
        var output = await answerer([TEXT, TEXT], [TEXT, TEXT]);
    }
    // (d) Batch input, top_k=2 -> QuestionAnsweringOutput[][]
    {
        var output = await answerer([TEXT, TEXT], [TEXT, TEXT], { top_k: 2 });
    }
}
// Summarization
{
    var summarizer = await (0, transformers_js_1.pipeline)("summarization", MODEL_ID);
    // (a) Single input -> SummarizationOutput
    {
        var output = await summarizer(TEXT);
    }
    // (b) Batch input -> SummarizationOutput
    {
        var output = await summarizer([TEXT, TEXT]);
    }
}
// Text Classification
{
    // Create a text classification pipeline
    var classifier = await (0, transformers_js_1.pipeline)("text-classification", MODEL_ID);
    // (a) Single input, top_k=1 -> TextClassificationOutput
    {
        var output = await classifier(TEXT, { top_k: 1 });
    }
    // (b) Single input, top_k=2 -> TextClassificationOutput
    {
        var output = await classifier(TEXT, { top_k: 2 });
    }
    // (c) Batch input, top_k=1 -> TextClassificationOutput
    {
        var output = await classifier([TEXT, TEXT], { top_k: 1 });
    }
    // (d) Batch input, top_k=2 -> TextClassificationOutput[]
    {
        var output = await classifier([TEXT, TEXT], { top_k: 2 });
    }
}
// Text Generation
{
    var generator = await (0, transformers_js_1.pipeline)("text-generation", MODEL_ID);
    // (a) Single input -> TextGenerationStringOutput
    {
        var output = await generator(TEXT);
    }
    // (b) Batch input -> TextGenerationStringOutput[]
    {
        var output = await generator([TEXT, TEXT]);
    }
    // (c) Chat input -> TextGenerationChatOutput
    {
        var output = await generator(MESSAGES);
    }
    // (d) Batch chat input -> TextGenerationChatOutput[]
    {
        var output = await generator([MESSAGES, MESSAGES]);
    }
    // (e) Chat input with generation parameters -> TextGenerationChatOutput
    {
        var output = await generator(MESSAGES, {
            max_new_tokens: 50,
            stopping_criteria: [new transformers_js_1.InterruptableStoppingCriteria()],
        });
    }
}
// Text-to-Audio
{
    var generator = await (0, transformers_js_1.pipeline)("text-to-audio", MODEL_ID);
    // (a) Single input -> RawAudio
    {
        var output = await generator(TEXT);
    }
    // (b) Batch input -> RawAudio[]
    {
        var output = await generator([TEXT, TEXT]);
    }
}
// Text2Text Generation
{
    var generator = await (0, transformers_js_1.pipeline)("text2text-generation", MODEL_ID);
    // (a) Single input -> Text2TextGenerationOutput
    {
        var output = await generator(TEXT);
    }
    // (b) Batch input -> Text2TextGenerationOutput
    {
        var output = await generator([TEXT, TEXT]);
    }
}
// Token Classification
{
    var classifier = await (0, transformers_js_1.pipeline)("token-classification", MODEL_ID);
    // (a) Single input -> TokenClassificationOutput
    {
        var output = await classifier(TEXT);
    }
    // (b) Batch input -> TokenClassificationOutput[]
    {
        var output = await classifier([TEXT, TEXT]);
    }
}
// Translation
{
    var translator = await (0, transformers_js_1.pipeline)("translation", MODEL_ID);
    // (a) Single input -> TranslationOutput
    {
        var output = await translator(TEXT);
    }
    // (b) Batch input -> TranslationOutput
    {
        var output = await translator([TEXT, TEXT]);
    }
}
// Zero-shot Audio Classification
{
    var classifier = await (0, transformers_js_1.pipeline)("zero-shot-audio-classification", MODEL_ID);
    // (a) Single input -> ZeroShotAudioClassificationOutput
    {
        var output = await classifier(FLOAT32, ["class A", "class B"]);
    }
    // (b) Batch input -> ZeroShotAudioClassificationOutput[]
    {
        var output = await classifier([FLOAT32, FLOAT32], ["class A", "class B"]);
    }
}
// Zero-shot Classification
{
    var classifier = await (0, transformers_js_1.pipeline)("zero-shot-classification", MODEL_ID);
    // (a) Single input -> ZeroShotClassificationOutput
    {
        var output = await classifier(TEXT, ["class A", "class B"]);
    }
    // (b) Batch input -> ZeroShotClassificationOutput[]
    {
        var output = await classifier([TEXT, TEXT], ["class A", "class B"]);
    }
}
// Zero-shot Image Classification
{
    var classifier = await (0, transformers_js_1.pipeline)("zero-shot-image-classification", MODEL_ID);
    // (a) Single input -> ZeroShotImageClassificationOutput
    {
        var output = await classifier(URL, ["class A", "class B"]);
    }
    // (b) Batch input -> ZeroShotImageClassificationOutput[]
    {
        var output = await classifier([URL, URL], ["class A", "class B"]);
    }
}
// Zero-shot Object Detection
{
    var detector = await (0, transformers_js_1.pipeline)("zero-shot-object-detection", MODEL_ID);
    // (a) Single input -> ZeroShotObjectDetectionOutput
    {
        var output = await detector(URL, ["class A", "class B"]);
    }
    // (b) Batch input -> ZeroShotObjectDetectionOutput[]
    {
        var output = await detector([URL, URL], ["class A", "class B"]);
    }
}
