// Main thread -> Worker
export const REQUEST = 'transformersjs_worker_pipeline';

// Worker -> Main thread
export const RESPONSE_READY = 'transformersjs_worker_ready';
export const RESPONSE_RESULT = 'transformersjs_worker_result';
export const RESPONSE_CALLBACK_INVOCATION = 'callback_bridge:invoke';
