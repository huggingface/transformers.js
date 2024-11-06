
import {
    env,
    AutoTokenizer,
    Moondream1ForConditionalGeneration,
    TextStreamer,
    StoppingCriteria,
    RawImage,
    AutoProcessor,
    Tensor,
    full,
} from '@xenova/transformers';

const DEVICE = 'webgpu';
const MAX_NEW_TOKENS = 256;

env.backends.onnx.wasm.proxy = DEVICE !== 'webgpu';

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
class TextGenerationPipeline {
    static model_id = 'Xenova/moondream2';
    static tokenizer = null;
    static processor = null;
    static model = null;
    static supportsFp16 = null;

    static async getInstance(progress_callback = null) {

        this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
            progress_callback,
        });

        this.processor ??= AutoProcessor.from_pretrained(this.model_id);

        // Choose the model based on whether fp16 is available
        this.supportsFp16 ??= await hasFp16();
        this.model ??= Moondream1ForConditionalGeneration.from_pretrained(this.model_id, {
            dtype: {
                embed_tokens: this.supportsFp16 ? 'fp16' : 'fp32', // or 'fp32'
                vision_encoder: this.supportsFp16 ? 'fp16' : 'fp32', // or 'q8'
                decoder_model_merged: 'q4', // or 'q4f16' or 'q8'
            },
            device: DEVICE,
            progress_callback,
        });

        return Promise.all([this.tokenizer, this.processor, this.model]);
    }
}


class CallbackTextStreamer extends TextStreamer {
    constructor(tokenizer, cb) {
        super(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
        });
        this.cb = cb;
    }

    on_finalized_text(text) {
        this.cb(text);
    }
}

class InterruptableStoppingCriteria extends StoppingCriteria {
    constructor() {
        super();
        this.interrupted = false;
    }

    interrupt() {
        this.interrupted = true;
    }

    reset() {
        this.interrupted = false;
    }

    _call(input_ids, scores) {
        return new Array(input_ids.length).fill(this.interrupted);
    }
}

const stopping_criteria = new InterruptableStoppingCriteria();

async function generate(messages) {

    // Only support a single image for now
    const images = messages.filter(x => x.image).map(x => x.image);
    if (images.length > 1) {
        self.postMessage({
            status: 'error',
            error: 'Currently, at most one image is supported.',
        });
        return;
    }

    // Retrieve the text-generation pipeline.
    const [tokenizer, processor, model] = await TextGenerationPipeline.getInstance();

    // Construct and tokenize prompt
    const prompt = messages.map(x => `${x.image ? '<image>\n\n' : ''}${x.role === 'user' ? 'Question: ' : 'Answer: '}${x.content.trim()}`).join('\n\n') + '\n\nAnswer:'
    let inputs = tokenizer(prompt);

    if (images.length > 0) {
        const image = await RawImage.fromURL(images[0]);
        const vision_inputs = await processor(image);

        inputs = { ...inputs, ...vision_inputs };
    }

    let startTime;
    let numTokens = 0;
    const cb = (output) => {
        startTime ??= performance.now();

        let tps;
        if (numTokens++ > 0) {
            tps = numTokens / (performance.now() - startTime) * 1000;
        }
        self.postMessage({
            status: 'update',
            output, tps, numTokens,
        });
    }

    const streamer = new CallbackTextStreamer(tokenizer, cb);

    // Tell the main thread we are starting
    self.postMessage({ status: 'start' });

    const outputs = await model.generate({
        ...inputs,
        max_new_tokens: MAX_NEW_TOKENS,
        streamer,
        stopping_criteria,
    });
    const outputText = tokenizer.batch_decode(outputs, { skip_special_tokens: false });

    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: outputText,
    });
}

async function load() {
    self.postMessage({
        status: 'loading',
        data: 'Loading model...'
    });

    // Load the pipeline and save it for future use.
    const [tokenizer, processor, model] = await TextGenerationPipeline.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    self.postMessage({
        status: 'loading',
        data: 'Compiling shaders and warming up model...'
    });

    // Run model with dummy input to compile shaders
    const text_inputs = tokenizer('a');

    const vision_inputs = {
        pixel_values: full([1, 3, 378, 378], 0.0)
    }

    const inputs = { ...text_inputs, ...vision_inputs };
    await model.generate({ ...inputs, max_new_tokens: 1 });
    self.postMessage({ status: 'ready' });
}
// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load();
            break;

        case 'generate':
            stopping_criteria.reset();
            generate(data);
            break;

        case 'interrupt':
            stopping_criteria.interrupt();
            break;

        case 'reset':
            stopping_criteria.reset();
            break;
    }
});
