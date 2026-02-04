/**
 * @file Constants for worker pipeline communication.
 * @module worker/constants
 */

/**
 * Message type for transformer pipeline operations in web workers.
 * @constant {string}
 */
export const REQUEST_MESSAGE_TYPE = 'transformersjs_worker_pipeline';

/**
 * Message type for invoking callbacks from worker to main thread.
 * @constant {string}
 */
export const RESPONSE_MESSAGE_TYPE_INVOKE_CALLBACK = 'transformersjs_worker_invokeCallback';

/**
 * Message type for pipeline ready notification.
 * @constant {string}
 */
export const RESPONSE_MESSAGE_TYPE_READY = 'transformersjs_worker_ready';

/**
 * Message type for pipeline result.
 * @constant {string}
 */
export const RESPONSE_MESSAGE_TYPE_RESULT = 'transformersjs_worker_result';
