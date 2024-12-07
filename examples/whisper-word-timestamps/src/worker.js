
import { pipeline } from '@xenova/transformers';

const PER_DEVICE_CONFIG = {
    webgpu: {
        dtype: {
            encoder_model: 'fp32',
            decoder_model_merged: 'q4',
        },
        device: 'webgpu',
    },
    wasm: {
        dtype: 'q8',
        device: 'wasm',
    },
};

/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
class PipelineSingeton {
    static model_id = 'onnx-community/whisper-base_timestamped';
    static instance = null;

    static async getInstance(progress_callback = null, device = 'webgpu') {

        if (!this.instance) {
            this.instance = pipeline('automatic-speech-recognition', this.model_id, {
                ...PER_DEVICE_CONFIG[device],
                progress_callback,
            });
        }
        return this.instance;
    }
}

async function load({ device }) {
    self.postMessage({
        status: 'loading',
        data: `Loading model (${device})...`
    });

    // Load the pipeline and save it for future use.
    const transcriber = await PipelineSingeton.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    }, device);

    if (device === 'webgpu') {
        self.postMessage({
            status: 'loading',
            data: 'Compiling shaders and warming up model...'
        });

        await transcriber(new Float32Array(16_000), {
            language: 'en',
        });
    }

    self.postMessage({ status: 'ready' });
}

async function run({ audio, language }) {
    const transcriber = await PipelineSingeton.getInstance();

    // Read and preprocess image
    const start = performance.now();

    const result = await transcriber(audio, {
        language,
        return_timestamps: 'word',
        chunk_length_s: 30,
    });

    const end = performance.now();

    self.postMessage({ status: 'complete', result, time: end - start });
}

// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load(data);
            break;

        case 'run':
            run(data);
            break;
    }
});
