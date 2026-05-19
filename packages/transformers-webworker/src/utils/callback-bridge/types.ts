/**
 * Represents a serialized function reference that can be sent across worker boundaries
 */
export interface SerializedFunction {
    __fn: true;
    functionId: string;
}

/**
 * Message sent from worker to main thread to invoke a callback
 */
export interface CallbackInvocationMessage {
    type: string;
    functionId: string;
    args: any[];
}

export type SerializableOptions = Record<string, any>;
export type SerializedOptions = Record<string, any>;
