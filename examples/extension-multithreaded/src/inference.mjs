import { pipeline, env } from '@xenova/transformers';

// Skip initial check for local models, since we are not loading any local models.
env.allowLocalModels = false;
// The extension sandbox page (which we need to use because of the way ONNX loads multiple threads)
// is prohibited from accesing the browser cache and will throw an erorr if it tries.
// Perhaps this can be mitigated with a custom cache instead of the browser cache?
env.useBrowserCache = false;
// env.backends.onnx.wasm.numThreads = 1;

class PipelineSingleton {
    static task = 'text-classification';
    static model = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { progress_callback });
        }

        return this.instance;
    }
}

export { PipelineSingleton }