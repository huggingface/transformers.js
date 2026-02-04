/**
 * @file Web Worker pipeline handler for managing transformer pipelines in a worker context.
 * @module worker/worker_pipeline_handler
 */

import { pipeline } from '../pipelines.js';
import {
    REQUEST_MESSAGE_TYPE,
    RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
    RESPONSE_MESSAGE_TYPE_READY,
    RESPONSE_MESSAGE_TYPE_RESULT,
} from './constants.js';

/**
 * Cache for storing initialized pipelines by their configuration.
 * @type {Map<string, any>}
 */
const pipelines = new Map();

/**
 * Creates a web worker pipeline handler that manages pipeline creation and execution.
 * @returns {{onmessage: (event: MessageEvent) => Promise<void>}} Handler object with onmessage method
 */
export const worker_pipeline_handler = () => {
    /**
     * Converts serialized options back to their original form.
     * Handles special cases like callback functions that were serialized with __fn marker.
     * @param {Record<string, any>} options - The options object to unserialize
     * @returns {Record<string, any>} The unserialized options object
     */
    const unserialize_options = (options) => {
        const out = {};
        Object.entries(options ?? {}).forEach(([key, value]) => {
            if (typeof value === 'object' && value && '__fn' in value && value.__fn) {
                out[key] = (...args) =>
                    self.postMessage({
                        type: RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
                        function_id: 'functionId' in value ? value.functionId : null,
                        args,
                    });
            } else {
                out[key] = value;
            }
        });
        return out;
    };

    return {
        /**
         * Message event handler for processing pipeline requests.
         * @param {MessageEvent} event - The message event from the main thread
         * @returns {Promise<void>}
         */
        onmessage: async (event) => {
            if (!event?.data || event.data?.type !== REQUEST_MESSAGE_TYPE) return;

            const { id, data, task, model_id, options, pipe_options = {} } = event.data;

            const key = JSON.stringify({ task, model_id, options });
            let pipe = pipelines.get(key);

            if (!pipe) {
                pipe = await pipeline(task, model_id, unserialize_options(options));
                pipelines.set(key, pipe);
            }

            self.postMessage({ id, type: RESPONSE_MESSAGE_TYPE_READY });

            const result = data ? await pipe(data, pipe_options) : null;

            self.postMessage({ id, type: RESPONSE_MESSAGE_TYPE_RESULT, result });
        },
    };
};
