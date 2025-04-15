
/////////////////////////////////////////////////////////////////
// Worker.js file for doing all transformer-based computations //
// Needed to ensure the UI thread is not blocked when running  //
/////////////////////////////////////////////////////////////////

import { pipeline, env, TextStreamer } from "@huggingface/transformers";
env.allowLocalModels = false;

// Define task function mapping
const TASK_FUNCTION_MAPPING = {
    'translation': translate,
    'text-generation': text_generation,
    'code-completion': code_completion,
    'masked-language-modelling': masked_lm,
    'sequence-classification': sequence_classification,
    'token-classification': token_classification,
    'zero-shot-classification': zero_shot_classification,
    'question-answering': question_answering,
    'summarization': summarize,
    'automatic-speech-recognition': speech_to_text,
    'image-to-text': image_to_text,
    'image-classification': image_classification,
    'zero-shot-image-classification': zero_shot_image_classification,
    'object-detection': object_detection,
}

// Listen for messages from UI
self.addEventListener('message', async (event) => {
    const data = event.data;
    let fn = TASK_FUNCTION_MAPPING[data.task];

    if (!fn) return;

    let result = await fn(data);
    self.postMessage({
        task: data.task,
        type: 'result',
        data: result
    });
});

// Define model factories
// Ensures only one model is created of each type
class PipelineFactory {
    static task = null;
    static model = null;

    // NOTE: instance stores a promise that resolves to the pipeline
    static instance = null;

    constructor(tokenizer, model) {
        this.tokenizer = tokenizer;
        this.model = model;
    }

    /**
     * Get pipeline instance
     * @param {*} progressCallback 
     * @returns {Promise}
     */
    static getInstance(progressCallback = null) {
        if (this.task === null || this.model === null) {
            throw Error("Must set task and model")
        }
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, {
                progress_callback: progressCallback
            });
        }

        return this.instance;
    }
}

class TranslationPipelineFactory extends PipelineFactory {
    static task = 'translation';
    static model = 'Xenova/t5-small';
}

class TextGenerationPipelineFactory extends PipelineFactory {
    static task = 'text-generation';
    static model = 'Xenova/distilgpt2';
}

class CodeCompletionPipelineFactory extends PipelineFactory {
    static task = 'text-generation';
    static model = 'Xenova/codegen-350M-mono';
}

class MaskedLMPipelineFactory extends PipelineFactory {
    static task = 'fill-mask';
    static model = 'Xenova/bert-base-cased';
}

class SequenceClassificationPipelineFactory extends PipelineFactory {
    static task = 'text-classification';
    static model = 'Xenova/bert-base-multilingual-uncased-sentiment';
}

class TokenClassificationPipelineFactory extends PipelineFactory {
    static task = 'token-classification';
    static model = 'Xenova/bert-base-multilingual-cased-ner-hrl';
}

class ZeroShotClassificationPipelineFactory extends PipelineFactory {
    static task = 'zero-shot-classification';
    static model = 'Xenova/distilbert-base-uncased-mnli';
}

class QuestionAnsweringPipelineFactory extends PipelineFactory {
    static task = 'question-answering';
    static model = 'Xenova/distilbert-base-cased-distilled-squad';
}

class SummarizationPipelineFactory extends PipelineFactory {
    static task = 'summarization';
    static model = 'Xenova/distilbart-cnn-6-6';
}

class AutomaticSpeechRecognitionPipelineFactory extends PipelineFactory {
    static task = 'automatic-speech-recognition';
    static model = 'Xenova/whisper-tiny.en';
}

class ImageToTextPipelineFactory extends PipelineFactory {
    static task = 'image-to-text';
    static model = 'Xenova/vit-gpt2-image-captioning';
}

class ImageClassificationPipelineFactory extends PipelineFactory {
    static task = 'image-classification';
    static model = 'Xenova/vit-base-patch16-224';
}


class ZeroShotImageClassificationPipelineFactory extends PipelineFactory {
    static task = 'zero-shot-image-classification';
    static model = 'Xenova/clip-vit-base-patch16';
}

class ObjectDetectionPipelineFactory extends PipelineFactory {
    static task = 'object-detection';
    static model = 'Xenova/detr-resnet-50';
}

async function translate(data) {

    let pipeline = await TranslationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'translation',
            data: data
        });
    })

    // Update task based on source and target languages
    // Doing it this way prevents the same model from being loaded multiple times
    pipeline.task = `translation_${data.languageFrom}_to_${data.languageTo}`;

    let allText = "";

    const streamer = new TextStreamer(pipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: function (newText) {
            allText += newText;

            self.postMessage({
                type: 'update',
                target: data.elementIdToUpdate,
                data: allText.trim()
            });
        }
    })

    return await pipeline(data.text, {
        ...data.generation,
        streamer
    })
}

async function text_generation(data) {

    let pipeline = await TextGenerationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'text-generation',
            data: data
        });
    })

    let text = data.text.trim();

    let allText = "";

    const streamer = new TextStreamer(pipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: function (newText) {
            allText += newText;

            self.postMessage({
                type: 'update',
                target: data.elementIdToUpdate,
                data: allText.trim()
            });
        }
    })

    return await pipeline(text, {
        ...data.generation,
        streamer
    })
}

async function code_completion(data) {

    let pipeline = await CodeCompletionPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'code-completion',
            data: data,
        });
    })

    let text = data.text;

    let allText = "";

    const streamer = new TextStreamer(pipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: function (newText) {
            allText += newText;

            self.postMessage({
                type: 'update',
                target: data.elementIdToUpdate,
                targetType: data.targetType,
                data: allText.trim()
            });
        }
    })

    return await pipeline(text, {
        ...data.generation,
        streamer
    })
}

async function masked_lm(data) {

    let pipeline = await MaskedLMPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'masked-language-modelling',
            data: data
        });
    })

    let output = await pipeline(data.text, data.generation)

    self.postMessage({
        type: 'update',
        target: data.elementIdToUpdate,
        data: output.map(x => x.sequence).join('\n')
    });

    return output;
}

async function sequence_classification(data) {

    let pipeline = await SequenceClassificationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'sequence-classification',
            data: data
        });
    });

    let outputs = await pipeline(data.text, {
        topk: 5 // return all
    })

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        targetType: data.targetType,
        data: outputs
    });
}


async function token_classification(data) {

    let pipeline = await TokenClassificationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'token-classification',
            data: data
        });
    });

    let outputs = await pipeline(data.text, {
        ignore_labels: []   // Return all labels
    });

    let chunks = [];
    let currentChunk = { type: '', text: [] };

    for (let i = 0; i < outputs.length; i++) {
        let word = pipeline.tokenizer.model.tokens_to_ids.get(outputs[i].word);
        let entity = outputs[i].entity;

        if (entity.startsWith('B-')) { // beginning of a new chunk
            if (currentChunk.text.length > 0) { // push the current chunk if it exists
                chunks.push(currentChunk);
                currentChunk = { type: '', text: [] };
            }
            currentChunk.type = entity.slice(2); // get the type of the chunk
            currentChunk.text = [word];
        } else if (entity.startsWith('I-')) { // continuation of a chunk
            currentChunk.text.push(word);
        } else { // not part of a chunk (O tag)
            if (currentChunk.text.length > 0) { // push the current chunk if it exists

                if (currentChunk.type === 'O') {
                    currentChunk.text.push(word);
                } else {
                    chunks.push(currentChunk);
                    currentChunk = { type: 'O', text: [word] };
                }
            } else {
                currentChunk = { type: 'O', text: [word] };
            }
        }
    }

    // push the last chunk if it exists
    if (currentChunk.text.length > 0) {
        chunks.push(currentChunk);
    }

    let postProcessedChunks = chunks.map(
        x => ({
            type: x.type,
            text: pipeline.tokenizer.decode(x.text)
        })
    )

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        targetType: data.targetType,
        data: postProcessedChunks,
    });
}


async function zero_shot_classification(data) {

    let pipeline = await ZeroShotClassificationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'zero-shot-classification',
            data: data
        });
    });

    let outputs = await pipeline(data.text, data.classes, data.generation);
    let formattedOutputs = outputs.labels.map((x, i) => {
        return {
            label: x,
            score: outputs.scores[i],
        }
    });

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        targetType: data.targetType,
        data: formattedOutputs
    });
}
async function question_answering(data) {

    let pipeline = await QuestionAnsweringPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'question-answering',
            data: data
        });
    })

    let answer = await pipeline(data.question, data.context)
    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        data: answer.answer
    });

    return answer;
}

async function summarize(data) {
    let pipeline = await SummarizationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'summarization',
            data: data
        });
    })

    let allText = "";

    const streamer = new TextStreamer(pipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: function (newText) {
            allText += newText;

            self.postMessage({
                type: 'update',
                target: data.elementIdToUpdate,
                data: allText
            });
        }
    })

    return await pipeline(data.text, {
        ...data.generation,
        streamer
    })
}

async function speech_to_text(data) {
    let pipeline = await AutomaticSpeechRecognitionPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'automatic-speech-recognition',
            data: data
        });
    })

    let allText = "";

    const streamer = new TextStreamer(pipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: function (newText) {
            allText += newText;

            self.postMessage({
                type: 'update',
                target: data.elementIdToUpdate,
                data: allText
            });
        }
    })

    return await pipeline(data.audio, {
        // Choose good defaults for the demo
        chunk_length_s: 30,
        stride_length_s: 5,

        ...data.generation,

        streamer
    })
}

async function image_to_text(data) {
    let pipeline = await ImageToTextPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'image-to-text',
            data: data
        });
    })

    let allText = "";

    const streamer = new TextStreamer(pipeline.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: function (newText) {
            allText += newText;

            self.postMessage({
                type: 'update',
                target: data.elementIdToUpdate,
                data: allText
            });
        }
    })

    return await pipeline(data.image, {
        ...data.generation,
        streamer
    })
}

async function image_classification(data) {
    let pipeline = await ImageClassificationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'image-classification',
            data: data
        });
    })

    let outputs = await pipeline(data.image, {
        topk: 5 // return all
    })

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        targetType: data.targetType,
        updateLabels: data.updateLabels,
        data: outputs
    });

}


async function zero_shot_image_classification(data) {
    let pipeline = await ZeroShotImageClassificationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'image-classification',
            data: data
        });
    })

    let outputs = await pipeline(data.image, data.classes)

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        targetType: data.targetType,
        updateLabels: data.updateLabels,
        data: outputs
    });

}


async function object_detection(data) {

    let pipeline = await ObjectDetectionPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'object-detection',
            data: data
        });
    })

    let outputs = await pipeline(data.image, {
        threshold: 0.9,
        percentage: true
    })

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        targetType: data.targetType,
        chartId: data.chartId,
        data: outputs
    });
}

