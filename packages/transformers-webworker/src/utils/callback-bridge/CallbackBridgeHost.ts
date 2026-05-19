import type { SerializedOptions, SerializableOptions, SerializedFunction } from './types.js';
import { RESPONSE_CALLBACK_INVOCATION } from '../../constants';

/**
 * Worker thread side of the callback bridge.
 * Deserializes callback references into functions that post messages back to main thread.
 */
export class CallbackBridgeHost {
    /**
     * Deserialize options by replacing callback references with proxy functions
     * that post messages back to the main thread
     */
    deserialize(options: SerializedOptions): SerializableOptions {
        const out: SerializableOptions = {};

        Object.entries(options ?? {}).forEach(([key, value]) => {
            if (this.isSerializedFunction(value)) {
                // Create a proxy function that posts a message to invoke the real callback
                out[key] = (...args: any[]) => {
                    self.postMessage({
                        type: RESPONSE_CALLBACK_INVOCATION,
                        functionId: value.functionId,
                        args,
                    });
                };
            } else {
                out[key] = value;
            }
        });

        return out;
    }

    /**
     * Type guard to check if a value is a serialized function
     */
    private isSerializedFunction(value: any): value is SerializedFunction {
        return (
            typeof value === 'object' &&
            value !== null &&
            '__fn' in value &&
            value.__fn === true &&
            'functionId' in value
        );
    }
}
