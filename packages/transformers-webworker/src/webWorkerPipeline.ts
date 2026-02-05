import { PipelineType } from '@huggingface/transformers';
import {
    REQUEST_MESSAGE_TYPE,
    RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
    RESPONSE_MESSAGE_TYPE_RESULT,
} from './constants.js';

const webWorkerPipeline = <PayloadType = any, ResultType = any>(
    worker: Worker,
    task: PipelineType,
    model_id: string,
    options: Record<string, any> = {},
) =>
    new Promise((resolve, reject) => {
        const callbackMap = new Map<string, Function>();

        const messagesResolversMap = new Map<number | 'init', { resolve: Function; reject: Function }>();
        let messageIdCounter = 0;

        const serializeOptions = (options: Record<string, any>) => {
            const out: Record<string, any> = {};
            Object.entries(options ?? {}).forEach(([key, value]) => {
                if (typeof value === 'function') {
                    const functionId = `cb_${key}`;
                    callbackMap.set(functionId, value);
                    out[key] = { __fn: true, functionId };
                } else {
                    out[key] = value;
                }
            });
            return out;
        };

        worker.onmessage = (e) => {
            const msg = e.data;
            if (msg?.type === RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK) {
                const { functionId, args } = msg;
                const fn = callbackMap.get(functionId);
                if (fn) {
                    fn(...args);
                }
            }
            if (msg?.type === RESPONSE_MESSAGE_TYPE_RESULT) {
                if (msg?.id === 'init') {
                    resolve((data: PayloadType, pipeOptions: Record<string, any>) => {
                        return new Promise<any>((resolve, reject) => {
                            const id = messageIdCounter++;
                            messagesResolversMap.set(id, { resolve, reject });
                            worker.postMessage({
                                id,
                                type: REQUEST_MESSAGE_TYPE,
                                data,
                                task,
                                model_id,
                                options: options ? serializeOptions(options) : {},
                                pipeOptions,
                            });
                        });
                    });
                } else {
                    const resolver = messagesResolversMap.get(msg.id);
                    if (resolver) {
                        if (msg.error) resolver.reject(msg.error);
                        else resolver.resolve(msg.result);
                        messagesResolversMap.delete(msg.id);
                    }
                }
            }
        };

        messagesResolversMap.set('init', { resolve, reject });
        worker.postMessage({
            id: 'init',
            type: REQUEST_MESSAGE_TYPE,
            data: null,
            task: task ?? '',
            model_id: model_id ?? '',
            options: options ? serializeOptions(options) : {},
        });
    });
export default webWorkerPipeline;
