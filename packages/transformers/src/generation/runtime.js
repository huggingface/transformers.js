import { Tensor } from '../utils/tensor.js';
import { createGenerationController } from './controller.js';

/**
 * @typedef {Object} GenerationCapabilitiesV1
 * @property {1} sessionVersion
 * @property {number} maxBatchSize
 * @property {string[]} cpuModes
 * @property {string[]} planModes
 * @property {boolean} cpuLogits
 * @property {string[]} declarativePlans
 * @property {{defaultDepth: number, maxDepth: number}} tokenPipeline
 * @property {boolean} customJavaScriptStoppingCriteria
 * @property {false} cacheReorder
 * @property {false} cacheExpand
 */

/**
 * @typedef {Object} LogitsLeaseV1
 * @property {1} version
 * @property {'float32'} dtype
 * @property {[number, number]} shape
 * @property {() => Promise<Float32Array>} read
 * @property {(plan: Object) => Promise<{tokenIds: Uint32Array, processedScores?: Float32Array}>} [select]
 * @property {() => void} release
 */

/**
 * @typedef {Object} AutoregressiveSessionV1
 * @property {1} version
 * @property {number} batchSize
 * @property {number} maxSequenceLength
 * @property {(inputs: Object) => Promise<LogitsLeaseV1>} prefill
 * @property {(inputs: Object) => Promise<LogitsLeaseV1>} decode
 * @property {(inputs: Object, plan: Object) => AsyncIterable<{tokenIds: Uint32Array, processedScores?: Float32Array}>} [generateWithPlan]
 * @property {() => Promise<void>} dispose
 */

/**
 * Install the Transformers.js-owned public generation method on a custom model.
 *
 * @param {Object|Function} model
 */
export function installGenerationRuntime(model) {
    const implementation = /** @type {any} */ (model);
    if (typeof implementation.createAutoregressiveSession === 'function') {
        implementation.generate = (options) => generateWithAutoregressiveSession(implementation, options);
    }
    return model;
}

/**
 * Run decoder-only generation through a pull-based custom runtime session.
 *
 * @param {Object} model
 * @param {Object} options
 */
export async function generateWithAutoregressiveSession(model, options) {
    const { input_ids, attention_mask = null, signal = undefined } = options;
    if (!(input_ids instanceof Tensor)) {
        throw new TypeError('Custom autoregressive generation requires an `input_ids` Tensor.');
    }
    if (model.config?.is_encoder_decoder) {
        throw new Error('Autoregressive session protocol version 1 only supports decoder-only models.');
    }

    const controller = createGenerationController(model, input_ids, options);
    if (controller.allDone) return controller.finalize();

    const capabilities = model.generationCapabilities;
    try {
        validateCapabilities(capabilities, controller, attention_mask);
        throwIfAborted(signal);
    } catch (error) {
        controller.abort(error);
        throw error;
    }
    const plan = controller.compileRuntimePlan(capabilities);
    if (!plan && !capabilities.cpuLogits) {
        const error = new Error(
            'This generation request requires CPU-visible logits, but the runtime does not support them.',
        );
        controller.abort(error);
        throw error;
    }

    /** @type {AutoregressiveSessionV1|null} */
    let session = null;
    /** @type {LogitsLeaseV1|null} */
    let lease = null;
    try {
        session = await model.createAutoregressiveSession({
            batchSize: controller.batchSize,
            maxSequenceLength: controller.maxSequenceLength,
            signal,
        });
        validateSession(session, controller);

        const prefillInputs = {
            inputIds: tensorToTokenBatch(input_ids),
            attentionMask: attention_mask ? tensorToAttentionMask(attention_mask) : undefined,
            signal,
        };
        if (plan && typeof session.generateWithPlan === 'function') {
            const decisions = session.generateWithPlan(prefillInputs, plan)[Symbol.asyncIterator]();
            try {
                while (true) {
                    const item = await decisions.next();
                    if (item.done) break;
                    throwIfAborted(signal);
                    if (controller.commit(item.value).allDone) break;
                }
            } finally {
                await decisions.return?.();
            }
            if (!controller.allDone) {
                throw new Error('Autoregressive runtime ended its generation plan before generation completed.');
            }
            return controller.finalize();
        }

        if (!capabilities.cpuLogits) {
            throw new Error(
                'This generation request requires CPU-visible logits, but the runtime does not support them.',
            );
        }

        lease = await session.prefill(prefillInputs);
        while (!controller.allDone) {
            throwIfAborted(signal);
            const currentLease = lease;
            lease = null;
            validateLease(currentLease, controller.batchSize);

            let values;
            try {
                values = await currentLease.read();
            } finally {
                currentLease.release();
            }
            if (!(values instanceof Float32Array)) {
                throw new TypeError('Logits lease `read()` must return a Float32Array.');
            }

            const step = await controller.step(new Tensor('float32', values, currentLease.shape));
            if (step.allDone) break;
            lease = await session.decode({
                tokenIds: tensorToTokenBatch(step.nextTokenIds),
                signal,
            });
        }
        return controller.finalize();
    } catch (error) {
        controller.abort(error);
        throw error;
    } finally {
        lease?.release();
        await session?.dispose();
    }
}

function validateCapabilities(capabilities, controller, attentionMask) {
    if (!capabilities || capabilities.sessionVersion !== 1) {
        throw new Error('Custom generation models must declare generation capabilities with `sessionVersion: 1`.');
    }
    if (controller.batchSize > capabilities.maxBatchSize) {
        throw new Error(
            `Runtime supports batch size ${capabilities.maxBatchSize}, but generation received ${controller.batchSize}.`,
        );
    }
    if (controller.generationConfig.num_return_sequences > 1) {
        throw new Error('Autoregressive session protocol version 1 does not support multiple return sequences.');
    }
    if (controller.generationConfig.num_beams > 1) {
        throw new Error('Autoregressive session protocol version 1 does not support beam search.');
    }
    if (controller.generationConfig.guidance_scale > 1) {
        throw new Error('Autoregressive session protocol version 1 does not support classifier-free guidance.');
    }
    if (controller.generationConfig.output_attentions || controller.generationConfig.output_hidden_states) {
        throw new Error('Autoregressive session protocol version 1 does not support attentions or hidden states.');
    }
    if (attentionMask && !isAllOnes(attentionMask)) {
        throw new Error('Autoregressive session protocol version 1 only supports absent or all-ones attention masks.');
    }

    const mode = controller.generationConfig.do_sample ? 'multinomial' : 'greedy';
    if (!capabilities.cpuModes?.includes(mode) && !capabilities.planModes?.includes(mode)) {
        throw new Error(`Runtime does not support ${mode} generation.`);
    }
    const tokenPipeline = capabilities.tokenPipeline;
    if (
        tokenPipeline &&
        (!Number.isInteger(tokenPipeline.defaultDepth) ||
            !Number.isInteger(tokenPipeline.maxDepth) ||
            tokenPipeline.defaultDepth < 1 ||
            tokenPipeline.defaultDepth > tokenPipeline.maxDepth)
    ) {
        throw new Error('Runtime declared an invalid token pipeline depth.');
    }
}

function validateSession(session, controller) {
    if (!session || session.version !== 1) throw new Error('Runtime returned an unsupported autoregressive session.');
    if (session.batchSize !== controller.batchSize) {
        throw new Error(`Runtime session batch size ${session.batchSize} does not match ${controller.batchSize}.`);
    }
    if (session.maxSequenceLength < controller.maxSequenceLength) {
        throw new Error(
            `Runtime session length ${session.maxSequenceLength} is less than required length ${controller.maxSequenceLength}.`,
        );
    }
    if (typeof session.dispose !== 'function') throw new Error('Autoregressive sessions must implement `dispose()`.');
}

function validateLease(lease, batchSize) {
    if (!lease || lease.version !== 1 || lease.dtype !== 'float32') {
        throw new Error('Runtime returned an unsupported logits lease.');
    }
    if (!Array.isArray(lease.shape) || lease.shape.length !== 2 || lease.shape[0] !== batchSize) {
        throw new Error('Logits lease must have shape [batch, vocabularySize].');
    }
    if (typeof lease.read !== 'function' || typeof lease.release !== 'function') {
        throw new Error('Logits lease must implement `read()` and `release()`.');
    }
}

function tensorToTokenBatch(tensor) {
    if (!(tensor instanceof Tensor) || tensor.dims.length !== 2) {
        throw new TypeError('Token IDs must be a rank-2 Tensor.');
    }
    const data = new Uint32Array(tensor.size);
    for (let index = 0; index < tensor.size; ++index) {
        const value = Number(tensor.data[index]);
        if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
            throw new RangeError(`Token ID at index ${index} is outside the uint32 range.`);
        }
        data[index] = value;
    }
    return { data, shape: /** @type {[number, number]} */ ([tensor.dims[0], tensor.dims[1]]) };
}

function tensorToAttentionMask(tensor) {
    if (!(tensor instanceof Tensor) || tensor.dims.length !== 2) {
        throw new TypeError('Attention mask must be a rank-2 Tensor.');
    }
    return {
        data: Uint8Array.from(tensor.data, Number),
        shape: /** @type {[number, number]} */ ([tensor.dims[0], tensor.dims[1]]),
    };
}

function isAllOnes(tensor) {
    return Array.from(tensor.data).every((value) => Number(value) === 1);
}

function throwIfAborted(signal) {
    if (!signal?.aborted) return;
    if (typeof signal.throwIfAborted === 'function') signal.throwIfAborted();
    throw signal.reason ?? new Error('Generation aborted.');
}
