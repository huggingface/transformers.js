import { Tensor } from '../utils/tensor.js';
import { DynamicCache } from '../cache_utils.js';
import { throwIfAborted } from '../utils/core.js';
import { createGenerationController } from './controller.js';

/**
 * @typedef {Object} SessionConcurrencyCapabilities
 * @property {number} maxActiveSessions
 * @property {1} concurrentOperationsPerSession
 */

/**
 * @typedef {Object} CausalGenerationCapabilitiesV1
 * @property {1} sessionVersion
 * @property {number} maxBatchSize
 * @property {ReadonlyArray<import('./controller.js').GenerationMode>} cpuModes
 * @property {ReadonlyArray<import('./controller.js').GenerationMode>} planModes
 * @property {boolean} cpuLogits
 * @property {ReadonlyArray<'argmax'|'multinomial'>} declarativePlans
 * @property {{readonly defaultDepth: number, readonly maxDepth: number}} tokenPipeline
 * @property {SessionConcurrencyCapabilities} [sessionConcurrency]
 */

/** @typedef {CausalGenerationCapabilitiesV1} GenerationCapabilitiesV1 */

/**
 * @typedef {Object} LogitsLeaseV1
 * @property {1} version
 * @property {'float32'} dtype
 * @property {readonly [number, number]} shape
 * @property {() => Promise<Float32Array>} read
 * @property {() => void} release
 */

/**
 * @typedef {Object} RuntimeTokenBatchV1
 * @property {Uint32Array} data
 * @property {readonly [number, number]} shape
 */

/**
 * @typedef {Object} RuntimeAttentionMaskV1
 * @property {Uint8Array} data
 * @property {readonly [number, number]} shape
 */

/**
 * @typedef {Object} AutoregressivePrefillInputsV1
 * @property {RuntimeTokenBatchV1} inputIds
 * @property {RuntimeAttentionMaskV1} [attentionMask]
 * @property {AbortSignal} [signal]
 */

/**
 * @typedef {Object} AutoregressiveDecodeInputsV1
 * @property {RuntimeTokenBatchV1} tokenIds
 * @property {AbortSignal} [signal]
 */

/**
 * @typedef {Object} RuntimeGenerationPlanV1
 * @property {1} version
 * @property {ReadonlyArray<{readonly op: 'token-mask', readonly getMask: (vocabSize: number) => Uint32Array}>} processors
 * @property {{readonly op: 'argmax'}|{readonly op: 'multinomial', readonly temperature: number, readonly topK: number}} sampler
 * @property {number} maxNewTokens
 * @property {number} [pipelineDepth]
 */

/**
 * @typedef {Object} RuntimeTokenDecisionV1
 * @property {Uint32Array} tokenIds
 * @property {Float32Array} [processedScores]
 */

/**
 * @typedef {Object} AutoregressiveSessionOptionsV1
 * @property {number} batchSize
 * @property {number} maxSequenceLength
 * @property {AbortSignal} [signal]
 */

/**
 * @typedef {Object} PlanAutoregressiveSessionV1
 * @property {1} version
 * @property {number} batchSize
 * @property {number} maxSequenceLength
 * @property {(inputs: AutoregressivePrefillInputsV1, plan: RuntimeGenerationPlanV1) => AsyncIterable<RuntimeTokenDecisionV1>} generateWithPlan
 * @property {() => Promise<void>} dispose
 */

/**
 * @typedef {Object} PullAutoregressiveSessionV1
 * @property {1} version
 * @property {number} batchSize
 * @property {number} maxSequenceLength
 * @property {(inputs: AutoregressivePrefillInputsV1) => Promise<LogitsLeaseV1>} prefill
 * @property {(inputs: AutoregressiveDecodeInputsV1) => Promise<LogitsLeaseV1>} decode
 * @property {(inputs: AutoregressivePrefillInputsV1, plan: RuntimeGenerationPlanV1) => AsyncIterable<RuntimeTokenDecisionV1>} [generateWithPlan]
 * @property {() => Promise<void>} dispose
 */

/** @typedef {PlanAutoregressiveSessionV1|PullAutoregressiveSessionV1} AutoregressiveSessionV1 */

/** @type {WeakMap<Object, number>} */
const activeSessionCounts = new WeakMap();

/**
 * Resolve the causal-generation capability while accepting the original flat V1 field.
 *
 * @param {Object|Function} model
 * @returns {CausalGenerationCapabilitiesV1|undefined}
 */
export function getCausalGenerationCapabilities(model) {
    return model?.capabilities?.causalGeneration ?? model?.generationCapabilities;
}

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
    const { inputs = null, input_ids = null, attention_mask = null, signal = undefined } = options;
    if (inputs !== null && input_ids !== null) {
        throw new TypeError('Custom autoregressive generation accepts either `inputs` or `input_ids`, but not both.');
    }
    const resolvedInputIds = input_ids ?? inputs;
    if (!(resolvedInputIds instanceof Tensor)) {
        throw new TypeError('Custom autoregressive generation requires an `inputs` or `input_ids` Tensor.');
    }
    if (model.config?.is_encoder_decoder) {
        throw new Error('Autoregressive session protocol version 1 only supports decoder-only models.');
    }

    const controller = createGenerationController(model, resolvedInputIds, options);
    const finalize = () =>
        controller.finalize(
            controller.generationConfig.return_dict_in_generate ? { past_key_values: new DynamicCache() } : {},
        );

    const capabilities = getCausalGenerationCapabilities(model);
    try {
        validateCapabilities(capabilities, controller, attention_mask);
        throwIfAborted(signal);
    } catch (error) {
        controller.abort(error);
        throw error;
    }
    if (controller.allDone) return finalize();
    const plan = controller.compileRuntimePlan(capabilities);
    const mode = controller.generationConfig.do_sample ? 'multinomial' : 'greedy';
    if (!plan && !capabilities.cpuLogits) {
        const error = new Error(
            'This generation request requires CPU-visible logits, but the runtime does not support them.',
        );
        controller.abort(error);
        throw error;
    }
    if (!plan && !capabilities.cpuModes.includes(mode)) {
        const error = new Error(`Runtime does not support ${mode} generation through its CPU logits path.`);
        controller.abort(error);
        throw error;
    }

    let releaseSessionSlot;
    try {
        releaseSessionSlot = acquireSessionSlot(model, capabilities);
    } catch (error) {
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
        validateSession(session, controller, plan !== null);

        const prefillInputs = {
            inputIds: tensorToTokenBatch(resolvedInputIds),
            attentionMask: attention_mask ? tensorToAttentionMask(attention_mask) : undefined,
            signal,
        };
        if (plan) {
            const planSession = /** @type {PlanAutoregressiveSessionV1} */ (session);
            const decisions = planSession.generateWithPlan(prefillInputs, plan)[Symbol.asyncIterator]();
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
            return finalize();
        }

        const pullSession = /** @type {PullAutoregressiveSessionV1} */ (session);
        lease = await pullSession.prefill(prefillInputs);
        while (!controller.allDone) {
            throwIfAborted(signal);
            const currentLease = lease;
            lease = null;
            let values;
            try {
                validateLease(currentLease, controller.batchSize);
                values = await currentLease.read();
            } finally {
                currentLease?.release?.();
            }
            if (!(values instanceof Float32Array)) {
                throw new TypeError('Logits lease `read()` must return a Float32Array.');
            }
            if (values.length !== currentLease.shape[0] * currentLease.shape[1]) {
                throw new Error('Logits lease data length does not match its declared shape.');
            }

            const step = await controller.step(new Tensor('float32', values, [...currentLease.shape]));
            if (step.allDone) break;
            lease = await pullSession.decode({
                tokenIds: tensorToTokenBatch(step.nextTokenIds),
                signal,
            });
        }
        return finalize();
    } catch (error) {
        controller.abort(error);
        throw error;
    } finally {
        try {
            lease?.release?.();
        } finally {
            try {
                await session?.dispose();
            } finally {
                releaseSessionSlot();
            }
        }
    }
}

function validateCapabilities(capabilities, controller, attentionMask) {
    if (!capabilities || capabilities.sessionVersion !== 1) {
        throw new Error('Custom generation models must declare generation capabilities with `sessionVersion: 1`.');
    }
    if (!Number.isInteger(capabilities.maxBatchSize) || capabilities.maxBatchSize < 1) {
        throw new Error('Runtime must declare a positive integer `maxBatchSize`.');
    }
    if (!Array.isArray(capabilities.cpuModes) || !Array.isArray(capabilities.planModes)) {
        throw new Error('Runtime must declare `cpuModes` and `planModes` arrays.');
    }
    if (typeof capabilities.cpuLogits !== 'boolean') {
        throw new Error('Runtime must declare whether CPU-visible logits are supported.');
    }
    if (!capabilities.cpuLogits && capabilities.cpuModes.length > 0) {
        throw new Error('Runtime cannot declare CPU generation modes when `cpuLogits` is false.');
    }
    if (!Array.isArray(capabilities.declarativePlans)) {
        throw new Error('Runtime must declare a `declarativePlans` array.');
    }
    if (capabilities.planModes.includes('greedy') && !capabilities.declarativePlans.includes('argmax')) {
        throw new Error('Runtime greedy plan mode requires the `argmax` declarative plan.');
    }
    if (capabilities.planModes.includes('multinomial') && !capabilities.declarativePlans.includes('multinomial')) {
        throw new Error('Runtime multinomial plan mode requires the `multinomial` declarative plan.');
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
        !tokenPipeline ||
        !Number.isInteger(tokenPipeline.defaultDepth) ||
        !Number.isInteger(tokenPipeline.maxDepth) ||
        tokenPipeline.defaultDepth < 1 ||
        tokenPipeline.defaultDepth > tokenPipeline.maxDepth
    ) {
        throw new Error('Runtime declared an invalid token pipeline depth.');
    }
    const concurrency = capabilities.sessionConcurrency;
    if (
        concurrency &&
        (!Number.isInteger(concurrency.maxActiveSessions) ||
            concurrency.maxActiveSessions < 1 ||
            concurrency.concurrentOperationsPerSession !== 1)
    ) {
        throw new Error('Runtime declared invalid session concurrency capabilities.');
    }
}

function validateSession(session, controller, usePlan) {
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
    if (usePlan && typeof session.generateWithPlan !== 'function') {
        throw new Error('Plan autoregressive sessions must implement `generateWithPlan()`.');
    }
    if (!usePlan && (typeof session.prefill !== 'function' || typeof session.decode !== 'function')) {
        throw new Error('Pull autoregressive sessions must implement `prefill()` and `decode()`.');
    }
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

function acquireSessionSlot(model, capabilities) {
    const limit = capabilities.sessionConcurrency?.maxActiveSessions;
    if (limit === undefined) return () => {};
    const active = activeSessionCounts.get(model) ?? 0;
    if (active >= limit) {
        throw new Error(`Runtime supports at most ${limit} active autoregressive session${limit === 1 ? '' : 's'}.`);
    }
    activeSessionCounts.set(model, active + 1);
    let released = false;
    return () => {
        if (released) return;
        released = true;
        const remaining = (activeSessionCounts.get(model) ?? 1) - 1;
        if (remaining > 0) activeSessionCounts.set(model, remaining);
        else activeSessionCounts.delete(model);
    };
}
