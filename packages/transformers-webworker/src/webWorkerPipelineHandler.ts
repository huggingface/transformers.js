import { pipeline } from '@huggingface/transformers';
import { REQUEST, RESPONSE_READY, RESPONSE_RESULT } from './constants.js';
import { CallbackBridgeHost } from './utils/callback-bridge';

const pipelines = new Map();

const webWorkerPipelineHandler = () => {
    const callbackBridge = new CallbackBridgeHost();

    return {
        onmessage: async (event: MessageEvent) => {
            if (!event?.data || event.data?.type !== REQUEST) return;
            const { id, data, task, model_id, options, pipeOptions = {} } = event.data;
            const key = JSON.stringify({ task, model_id, options });
            let pipe = pipelines.get(key);
            if (!pipe) {
                pipe = await pipeline(task, model_id, callbackBridge.deserialize(options));
                pipelines.set(key, pipe);
            }
            self.postMessage({ id, type: RESPONSE_READY });
            const result = data ? await pipe(data, pipeOptions) : null;
            self.postMessage({ id, type: RESPONSE_RESULT, result });
        },
    };
};

export default webWorkerPipelineHandler;
