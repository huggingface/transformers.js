import type { CallbackInvocationMessage, SerializableOptions, SerializedOptions } from './types.js';
import { RESPONSE_CALLBACK_INVOCATION } from '../../constants';

/**
 * Main thread side of the callback bridge.
 * Serializes functions into callback references and handles invocations from the worker.
 */
export class CallbackBridgeClient {
    private callbackMap = new Map<string, Function>();
    private worker: Worker;
    private messageHandler: (event: MessageEvent) => void;

    constructor(worker: Worker) {
        this.worker = worker;
        this.messageHandler = this.handleMessage.bind(this);
        this.worker.addEventListener('message', this.messageHandler);
    }

    /**
     * Serialize options by replacing functions with callback references.
     * Non-serializable values (e.g. certain GPU devices or typed arrays in `session_options`)
     * are detected and logged as warnings but do not cause this method to throw.
     */
    serialize(options: SerializableOptions): SerializedOptions {
        const out: SerializedOptions = {};

        Object.entries(options ?? {}).forEach(([key, value]) => {
                const functionId = `cb_${key}_${Date.now()}_${Math.random().toString(36).slice(2)}`;
                this.callbackMap.set(functionId, value);
                out[key] = { __fn: true, functionId };
            } else {
                // Validate session_options for non-serializable objects
                if (key === 'session_options' && value) {
                    this.validateSessionOptions(value);
                }
                out[key] = value;
            }
        });

        return out;
    }

    /**
     * Validate session_options for non-serializable objects
     * @private
     */
    private validateSessionOptions(sessionOptions: any): void {
        if (typeof sessionOptions !== 'object' || sessionOptions === null) {
            return;
        }

        // Check for GPU device objects in executionProviders
        if (Array.isArray(sessionOptions.executionProviders)) {
            for (const provider of sessionOptions.executionProviders) {
                if (typeof provider === 'object' && provider !== null) {
                    // Check for GPUDevice
                    if ('device' in provider && provider.device !== null && typeof provider.device === 'object') {
                        console.warn(
                            'CallbackBridgeClient: session_options.executionProviders contains a GPUDevice object which cannot be serialized across worker boundaries. ' +
                                'Use the "device" parameter instead (e.g., device: "webgpu") and let the worker create its own GPU context.',
                        );
                    }
                    // Check for MLContext (WebNN)
                    if ('context' in provider && provider.context !== null && typeof provider.context === 'object') {
                        console.warn(
                            'CallbackBridgeClient: session_options.executionProviders contains an MLContext object which cannot be serialized across worker boundaries. ' +
                                'Use the "device" parameter instead (e.g., device: "webnn") and let the worker create its own WebNN context.',
                        );
                    }
                    // Check for gpuDevice in WebNN options
                    if (
                        'gpuDevice' in provider &&
                        provider.gpuDevice !== null &&
                        typeof provider.gpuDevice === 'object'
                    ) {
                        console.warn(
                            'CallbackBridgeClient: session_options.executionProviders contains a gpuDevice object which cannot be serialized across worker boundaries. ' +
                                'Use the "device" parameter instead and let the worker create its own GPU context.',
                        );
                    }
                }
            }
        }

        // Check for external data with typed arrays
        if (Array.isArray(sessionOptions.externalData)) {
            for (const external of sessionOptions.externalData) {
                if (
                    external?.data &&
                    (external.data instanceof Uint8Array ||
                        external.data instanceof ArrayBuffer ||
                        ArrayBuffer.isView(external.data))
                ) {
                    console.warn(
                        'CallbackBridgeClient: session_options.externalData contains typed arrays or ArrayBuffers. ' +
                            'Models with external data files should be loaded directly in the worker thread, not configured via session_options from the main thread. ' +
                            'The worker will automatically handle external data when loading large models.',
                    );
                    break;
                }
            }
        }
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
                fn(...args);
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
