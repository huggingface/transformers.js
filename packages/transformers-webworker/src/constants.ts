// Main thread -> Worker
export const REQUEST = 'transformersjs_worker_pipeline';

// Worker -> Main thread
export const RESPONSE_RESULT = 'transformersjs_worker_result';
export const RESPONSE_CALLBACK_INVOCATION = 'callback_bridge:invoke';

export const OPERATION_INIT = 'init';
export const OPERATION_RUN = 'run';
export const OPERATION_DISPOSE = 'dispose';
