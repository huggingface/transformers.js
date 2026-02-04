/**
 * @file Web Worker pipeline wrapper for executing pipelines in a worker thread.
 * @module worker/worker_pipeline
 */

import {
    REQUEST_MESSAGE_TYPE,
    RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK,
    RESPONSE_MESSAGE_TYPE_RESULT,
} from './constants.js';

/**
 * @typedef {import('../pipelines.js').PipelineType} PipelineType
 */

/**
 * Creates a pipeline that runs in a Web Worker.
 * @template PayloadType
 * @template ResultType
 * @param {Worker} worker - The Web Worker instance to use for pipeline execution
 * @param {PipelineType} task - The pipeline task type
 * @param {string} model_id - The model identifier to load
 * @param {Record<string, any>} [options={}] - Options for pipeline initialization
 * @returns {Promise<(data: PayloadType, pipe_options: Record<string, any>) => Promise<ResultType>>} A function that executes the pipeline
 */
export const worker_pipeline = (worker, task, model_id, options = {}) =>
    new Promise((resolve, reject) => {
        /**
         * Map storing callback functions by their ID.
         * @type {Map<string, Function>}
         */
        const callback_map = new Map();

        /**
         * Map storing promise resolvers/rejecters for each message.
         * @type {Map<number | "init", { resolve: Function; reject: Function }>}
         */
        const messages_resolvers_map = new Map();

        /**
         * Counter for generating unique message IDs.
         * @type {number}
         */
        let message_id_counter = 0;

        /**
         * Serializes options, converting functions to references that can be invoked via postMessage.
         * @param {Record<string, any>} options - The options object to serialize
         * @returns {Record<string, any>} The serialized options object
         */
        const serialize_options = (options) => {
            const out = {};
            Object.entries(options ?? {}).forEach(([key, value]) => {
                if (typeof value === 'function') {
                    const function_id = `cb_${key}`;
                    callback_map.set(function_id, value);
                    out[key] = { __fn: true, functionId: function_id };
                } else {
                    out[key] = value;
                }
            });
            return out;
        };

        /**
         * Message event handler for processing worker responses.
         * @param {MessageEvent} e - The message event from the worker
         */
        worker.onmessage = (e) => {
            const msg = e.data;

            // Handle callback invocations from the worker
            if (msg?.type === RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK) {
                const { functionId, args } = msg;
                const fn = callback_map.get(functionId);
                if (fn) {
                    fn(...args);
                }
            }

            // Handle result messages
            if (msg?.type === RESPONSE_MESSAGE_TYPE_RESULT) {
                if (msg?.id === 'init') {
                    // Initial setup complete - resolve with the pipeline function
                    resolve((data, pipe_options) => {
                        return new Promise((resolve, reject) => {
                            const id = message_id_counter++;
                            messages_resolvers_map.set(id, { resolve, reject });
                            worker.postMessage({
                                id,
                                type: REQUEST_MESSAGE_TYPE,
                                data,
                                task,
                                model_id,
                                options: options ? serialize_options(options) : {},
                                pipe_options,
                            });
                        });
                    });
                } else {
                    // Regular pipeline execution result
                    const resolver = messages_resolvers_map.get(msg.id);
                    if (resolver) {
                        if (msg.error) resolver.reject(msg.error);
                        else resolver.resolve(msg.result);
                        messages_resolvers_map.delete(msg.id);
                    }
                }
            }
        };

        // Initialize the pipeline in the worker
        messages_resolvers_map.set('init', { resolve, reject });
        worker.postMessage({
            id: 'init',
            type: REQUEST_MESSAGE_TYPE,
            data: null,
            task: task ?? '',
            model_id: model_id ?? '',
            options: options ? serialize_options(options) : {},
        });
    });
