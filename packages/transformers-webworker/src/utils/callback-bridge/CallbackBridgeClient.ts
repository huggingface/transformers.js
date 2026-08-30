import type { CallbackInvocationMessage, SerializableOptions, SerializedOptions } from './types';
import { RESPONSE_CALLBACK_INVOCATION } from '../../constants';

/**
 * Main thread side of the callback bridge.
 * Serializes functions into callback references and handles invocations from the worker.
 */
export class CallbackBridgeClient {
    private callbackMap = new Map<string, Function>();
    private callbackIdCounter = 0;
    private worker: Worker;
    private messageHandler: (event: MessageEvent) => void;

    constructor(worker: Worker) {
        this.worker = worker;
        this.messageHandler = this.handleMessage.bind(this);
        this.worker.addEventListener('message', this.messageHandler);
    }

    /**
     * Serialize options by replacing functions with callback references
     */
    serialize(options: SerializableOptions): { options: SerializedOptions; callbackIds: string[] } {
        const out: SerializedOptions = {};
        const callbackIds: string[] = [];

        Object.entries(options ?? {}).forEach(([key, value]) => {
            if (typeof value === 'function') {
                const functionId = `cb_${this.callbackIdCounter++}`;
                this.callbackMap.set(functionId, value);
                callbackIds.push(functionId);
                out[key] = { __fn: true, functionId };
            } else {
                out[key] = value;
            }
        });

        return { options: out, callbackIds };
    }

    /**
     * Handle incoming callback invocation messages from the worker
     */
    private handleMessage(event: MessageEvent): void {
        const msg = event.data as CallbackInvocationMessage;

        if (msg?.type === RESPONSE_CALLBACK_INVOCATION) {
            const { functionId, args } = msg;
            const fn = this.callbackMap.get(functionId);

            if (fn) {
                try {
                    fn(...(Array.isArray(args) ? args : []));
                } catch (error) {
                    console.error('CallbackBridgeClient: Callback invocation failed', error);
                }
            } else {
                console.warn(`CallbackBridgeClient: Unknown callback function ID: ${functionId}`);
            }
        }
    }

    /**
     * Clear a specific callback by its ID
     */
    clearCallback(functionId: string): void {
        this.callbackMap.delete(functionId);
    }

    /** Clear callbacks owned by a disposed pipeline. */
    clearCallbacks(functionIds: string[]): void {
        for (const functionId of functionIds) {
            this.callbackMap.delete(functionId);
        }
    }

    /**
     * Clear all callbacks
     */
    clearAllCallbacks(): void {
        this.callbackMap.clear();
    }

    /**
     * Clean up and remove event listeners
     */
    dispose(): void {
        this.worker.removeEventListener('message', this.messageHandler);
        this.callbackMap.clear();
    }
}
