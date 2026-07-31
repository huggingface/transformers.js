import type { PipelineType, PretrainedModelOptions } from '@huggingface/transformers';
import { OPERATION_DISPOSE, OPERATION_INIT, OPERATION_RUN, REQUEST, RESPONSE_RESULT } from './constants';
import { CallbackBridgeClient } from './utils/callback-bridge';

type PipelineCallOptions = Record<string, any>;

export type WebWorkerPipeline<
    PayloadType,
    ResultType,
    PipeOptions extends PipelineCallOptions = PipelineCallOptions,
> = {
    (data: PayloadType, pipeOptions?: PipeOptions): Promise<ResultType>;
    dispose(): Promise<void>;
};

interface PendingRequest {
    resolve: (value: any) => void;
    reject: (reason?: any) => void;
}

interface WorkerState {
    callbackBridge: CallbackBridgeClient;
    nextRequestId: number;
    nextPipelineId: number;
    pendingRequests: Map<number, PendingRequest>;
}

const workerStates = new WeakMap<Worker, WorkerState>();

const deserializeError = (value: any): Error => {
    const error = new Error(value?.message ?? String(value ?? 'Web Worker pipeline request failed'));
    if (value?.name) error.name = value.name;
    if (value?.stack) error.stack = value.stack;
    return error;
};

const getWorkerState = (worker: Worker): WorkerState => {
    const existingState = workerStates.get(worker);
    if (existingState) return existingState;

    const state: WorkerState = {
        callbackBridge: new CallbackBridgeClient(worker),
        nextRequestId: 0,
        nextPipelineId: 0,
        pendingRequests: new Map(),
    };

    worker.addEventListener('message', (event) => {
        const message = event.data;
        if (message?.type !== RESPONSE_RESULT || typeof message.id !== 'number') return;

        const pendingRequest = state.pendingRequests.get(message.id);
        if (!pendingRequest) return;

        state.pendingRequests.delete(message.id);
        if (message.error) pendingRequest.reject(deserializeError(message.error));
        else pendingRequest.resolve(message.result);
    });

    const rejectPendingRequests = (event: Event) => {
        const reason = (event as ErrorEvent).error ?? new Error('Web Worker failed');
        for (const request of state.pendingRequests.values()) request.reject(reason);
        state.pendingRequests.clear();
    };
    worker.addEventListener('error', rejectPendingRequests);
    worker.addEventListener('messageerror', rejectPendingRequests);

    workerStates.set(worker, state);
    return state;
};

const sendRequest = (worker: Worker, state: WorkerState, message: Record<string, any>): Promise<any> => {
    const id = state.nextRequestId++;
    return new Promise((resolve, reject) => {
        state.pendingRequests.set(id, { resolve, reject });
        try {
            worker.postMessage({ ...message, id, type: REQUEST });
        } catch (error) {
            state.pendingRequests.delete(id);
            reject(error);
        }
    });
};

const webWorkerPipeline = async <
    PayloadType = any,
    ResultType = any,
    PipeOptions extends PipelineCallOptions = PipelineCallOptions,
>(
    worker: Worker,
    task: PipelineType,
    model_id: string,
    options: PretrainedModelOptions = {},
): Promise<WebWorkerPipeline<PayloadType, ResultType, PipeOptions>> => {
    const state = getWorkerState(worker);
    const pipelineId = `pipeline_${state.nextPipelineId++}`;
    const serializedOptions = state.callbackBridge.serialize(options);

    try {
        await sendRequest(worker, state, {
            operation: OPERATION_INIT,
            pipelineId,
            task,
            model_id,
            options: serializedOptions.options,
        });
    } catch (error) {
        state.callbackBridge.clearCallbacks(serializedOptions.callbackIds);
        throw error;
    }

    let disposed = false;
    let disposePromise: Promise<void> | null = null;
    const pipe = ((data: PayloadType, pipeOptions: PipeOptions = {} as PipeOptions) => {
        if (disposed) return Promise.reject(new Error('Web Worker pipeline has been disposed'));
        return sendRequest(worker, state, {
            operation: OPERATION_RUN,
            pipelineId,
            data,
            pipeOptions,
        });
    }) as WebWorkerPipeline<PayloadType, ResultType, PipeOptions>;

    pipe.dispose = () => {
        if (disposePromise) return disposePromise;
        disposed = true;
        disposePromise = sendRequest(worker, state, { operation: OPERATION_DISPOSE, pipelineId }).then(
            () => state.callbackBridge.clearCallbacks(serializedOptions.callbackIds),
            (error) => {
                disposed = false;
                disposePromise = null;
                throw error;
            },
        );
        return disposePromise;
    };

    return pipe;
};

export default webWorkerPipeline;
