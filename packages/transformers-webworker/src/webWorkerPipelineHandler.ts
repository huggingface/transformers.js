import { pipeline } from '@huggingface/transformers';
import { OPERATION_DISPOSE, OPERATION_INIT, OPERATION_RUN, REQUEST, RESPONSE_RESULT } from './constants';
import { CallbackBridgeHost } from './utils/callback-bridge';

const serializeError = (error: any) => ({
    name: error?.name ?? 'Error',
    message: error?.message ?? String(error),
    stack: error?.stack,
});

const webWorkerPipelineHandler = () => {
    const callbackBridge = new CallbackBridgeHost();
    const pipelines = new Map<string, any>();
    const pipelineQueues = new Map<string, Promise<void>>();

    const processRequest = async (message: any) => {
        const { id, operation, pipelineId } = message;
        try {
            if (typeof id !== 'number' || typeof pipelineId !== 'string') {
                throw new TypeError('Invalid Web Worker pipeline request');
            }

            let result = null;
            if (operation === OPERATION_INIT) {
                if (pipelines.has(pipelineId)) {
                    throw new Error(`Web Worker pipeline already exists: ${pipelineId}`);
                }
                const { task, model_id, options = {} } = message;
                const pipe = await pipeline(task, model_id, callbackBridge.deserialize(options));
                pipelines.set(pipelineId, pipe);
            } else if (operation === OPERATION_RUN) {
                const pipe = pipelines.get(pipelineId);
                if (!pipe) throw new Error(`Unknown Web Worker pipeline: ${pipelineId}`);
                result = await pipe(message.data, message.pipeOptions ?? {});
            } else if (operation === OPERATION_DISPOSE) {
                const pipe = pipelines.get(pipelineId);
                if (pipe) {
                    await pipe.dispose?.();
                    pipelines.delete(pipelineId);
                }
            } else {
                throw new TypeError(`Unknown Web Worker pipeline operation: ${operation}`);
            }

            self.postMessage({ id, type: RESPONSE_RESULT, result });
        } catch (error) {
            self.postMessage({ id, type: RESPONSE_RESULT, error: serializeError(error) });
        }
    };

    return {
        onmessage: async (event: MessageEvent) => {
            if (!event?.data || event.data.type !== REQUEST) return;

            const { pipelineId } = event.data;
            if (typeof pipelineId !== 'string') return processRequest(event.data);

            const previousRequest = pipelineQueues.get(pipelineId) ?? Promise.resolve();
            const currentRequest = previousRequest.catch(() => undefined).then(() => processRequest(event.data));
            pipelineQueues.set(pipelineId, currentRequest);
            await currentRequest;
            if (pipelineQueues.get(pipelineId) === currentRequest) pipelineQueues.delete(pipelineId);
        },
    };
};

export default webWorkerPipelineHandler;
