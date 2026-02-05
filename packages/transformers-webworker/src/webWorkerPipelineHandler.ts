import { pipeline } from '@huggingface/transformers';
import {
    REQUEST_MESSAGE_TYPE,
    RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
    RESPONSE_MESSAGE_TYPE_READY,
    RESPONSE_MESSAGE_TYPE_RESULT,
} from './constants.js';

const pipelines = new Map();

const webWorkerPipelineHandler = () => {
    const unserializeOptions = (options: Record<string, any>) => {
        const out: Record<string, any> = {};
        Object.entries(options ?? {}).forEach(([key, value]) => {
            if (typeof value === 'object' && value && '__fn' in value && value.__fn) {
                out[key] = (...args: Array<any>) =>
                    self.postMessage({
                        type: RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
                        functionId: 'functionId' in value ? value.functionId : null,
                        args,
                    });
            } else {
                out[key] = value;
            }
        });
        return out;
    };

    return {
        onmessage: async (event: MessageEvent) => {
            if (!event?.data || event.data?.type !== REQUEST_MESSAGE_TYPE) return;
            const { id, data, task, model_id, options, pipeOptions = {} } = event.data;
            const key = JSON.stringify({ task, model_id, options });
            let pipe = pipelines.get(key);
            if (!pipe) {
                pipe = await pipeline(task, model_id, unserializeOptions(options));
                pipelines.set(key, pipe);
            }
            self.postMessage({ id, type: RESPONSE_MESSAGE_TYPE_READY });
            const result = data ? await pipe(data, pipeOptions) : null;
            self.postMessage({ id, type: RESPONSE_MESSAGE_TYPE_RESULT, result });
        },
    };
};

export default webWorkerPipelineHandler;
