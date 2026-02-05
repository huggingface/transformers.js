import { PipelineType } from '@huggingface/transformers';
import { REQUEST, RESPONSE_RESULT } from './constants.js';
import { CallbackBridgeClient } from './utils/callback-bridge';

const webWorkerPipeline = <PayloadType = any, ResultType = any>(
    worker: Worker,
    task: PipelineType,
    model_id: string,
    options: Record<string, any> = {},
) =>
    new Promise((resolve, reject) => {
        const callbackBridge = new CallbackBridgeClient(worker);

        const messagesResolversMap = new Map<number | 'init', { resolve: Function; reject: Function }>();
        let messageIdCounter = 0;

        const originalOnMessage = worker.onmessage;
        worker.onmessage = (e) => {
            const msg = e.data;
            if (msg?.type === RESPONSE_RESULT) {
                if (msg?.id === 'init') {
                    resolve((data: PayloadType, pipeOptions: Record<string, any>) => {
                        return new Promise<any>((resolve, reject) => {
                            const id = messageIdCounter++;
                            messagesResolversMap.set(id, { resolve, reject });
                            worker.postMessage({
                                id,
                                type: REQUEST,
                                data,
                                task,
                                model_id,
                                options: options ? callbackBridge.serialize(options) : {},
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
            type: REQUEST,
            data: null,
            task: task ?? '',
            model_id: model_id ?? '',
            options: options ? callbackBridge.serialize(options) : {},
        });
    });
export default webWorkerPipeline;
