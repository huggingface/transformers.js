
import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    RawImage,
    full,
} from '@xenova/transformers';

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has('shader-f16');
    } catch (e) {
        return false;
    }
}

/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
class Florence2Singleton {
    static model_id = 'onnx-community/Florence-2-base-ft';

    static async getInstance(progress_callback = null) {
        this.processor ??= AutoProcessor.from_pretrained(this.model_id);
        this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id);

        this.supports_fp16 ??= await hasFp16();
        this.model ??= Florence2ForConditionalGeneration.from_pretrained(this.model_id, {
            dtype: {
                embed_tokens: this.supports_fp16 ? 'fp16' : 'fp32',
                vision_encoder: this.supports_fp16 ? 'fp16' : 'fp32',
                encoder_model: 'q4', // or 'fp16' or 'fp32'
                decoder_model_merged: 'q4', // or 'fp16' or 'fp32'
            },
            device: 'webgpu',
            progress_callback,
        });

        return Promise.all([this.model, this.tokenizer, this.processor]);
    }
}


async function load() {
    self.postMessage({
        status: 'loading',
        data: 'Loading model...'
    });

    // Load the pipeline and save it for future use.
    const [model, tokenizer, processor] = await Florence2Singleton.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    self.postMessage({
        status: 'loading',
        data: 'Compiling shaders and warming up model...'
    });

    // Dummy text and vision inputs
    const text_inputs = tokenizer('a');
    const pixel_values = full([1, 3, 768, 768], 0.0);

    // Run model with dummy input to compile shaders
    await model.generate({
        ...text_inputs,
        pixel_values,
        max_new_tokens: 1,
    });

    self.postMessage({ status: 'ready' });
}

const TASKS_WITH_INPUTS = [
    '<CAPTION_TO_PHRASE_GROUNDING>',
]

let vision_inputs;
let image_size;
async function run({ text, url, task }) {
    const [model, tokenizer, processor] = await Florence2Singleton.getInstance();

    // Read and preprocess image
    const start = performance.now();
    if (!vision_inputs) {
        // Cache vision inputs when possible
        const image = await RawImage.fromURL(url);
        image_size = image.size;
        vision_inputs = await processor(image);
    }

    let user_input = task;
    if (TASKS_WITH_INPUTS.includes(task) && text) {
        user_input += text;
    }
    const prompts = processor.construct_prompts(user_input);
    const text_inputs = tokenizer(prompts);

    // Generate text
    const generated_ids = await model.generate({
        ...text_inputs,
        ...vision_inputs,
        max_new_tokens: 128,
        num_beams: 1,
        do_sample: false,
    });

    // Decode generated text
    const generated_text = tokenizer.batch_decode(generated_ids, { skip_special_tokens: false })[0];

    // Post-process the generated text
    const result = processor.post_process_generation(generated_text, task, image_size);

    const end = performance.now();

    self.postMessage({ status: 'complete', result, time: end - start });
}

// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load();
            break;

        case 'run':
            run(data);
            break;

        case 'reset':
            vision_inputs = image_size = null;
            break;
    }
});
