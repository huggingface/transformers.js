import { MODEL_MAPPING_NAMES, MODEL_TYPE_MAPPING } from '../../models/modeling_utils.js';
import { MODEL_TYPES } from '../../models/session_config.js';
import { GITHUB_ISSUE_URL } from '../constants.js';
import { logger } from '../logger.js';

/**
 * @typedef {import('../../configs.js').PretrainedConfig} PretrainedConfig
 */

/**
 * Resolves the model type (e.g., EncoderOnly, DecoderOnly, Seq2Seq, …) from a
 * model config by checking architectures and model_type against the known
 * MODEL_TYPE_MAPPING.
 *
 * Resolution order:
 *   1. `config.architectures` entries looked up in MODEL_TYPE_MAPPING
 *   2. `config.model_type` looked up directly in MODEL_TYPE_MAPPING
 *   3. `config.model_type` looked up via MODEL_MAPPING_NAMES → architecture → MODEL_TYPE_MAPPING
 *   4. Fallback to `MODEL_TYPES.EncoderOnly`
 *
 * @param {PretrainedConfig} config The model config object.
 * @param {{ warn?: boolean }} [options] Set `warn` to false to suppress the
 *   fallback warning (defaults to true).
 * @returns {number} One of the MODEL_TYPES enum values.
 */
export function resolve_model_type(config, { warn = true } = {}) {
    // @ts-ignore - architectures is set via Object.assign in PretrainedConfig constructor
    const architectures = /** @type {string[]} */ (config.architectures || []);

    // 1. Try architectures against MODEL_TYPE_MAPPING
    for (const arch of architectures) {
        const mappedType = MODEL_TYPE_MAPPING.get(arch);
        if (mappedType !== undefined) {
            return mappedType;
        }
    }

    // 2. Try config.model_type directly
    if (config.model_type) {
        const mappedType = MODEL_TYPE_MAPPING.get(config.model_type);
        if (mappedType !== undefined) {
            return mappedType;
        }

        // 3. Try MODEL_MAPPING_NAMES as a last resort
        for (const mapping of Object.values(MODEL_MAPPING_NAMES)) {
            if (mapping.has(config.model_type)) {
                const resolved = MODEL_TYPE_MAPPING.get(mapping.get(config.model_type));
                if (resolved !== undefined) {
                    return resolved;
                }
            }
        }
    }

    // 4. Fallback
    if (warn) {
        const archList = architectures.length > 0 ? architectures.join(', ') : '(none)';
        logger.warn(
            `[resolve_model_type] Architecture(s) not found in MODEL_TYPE_MAPPING: [${archList}] ` +
                `for model type '${config.model_type}'. Falling back to EncoderOnly (single model.onnx file). ` +
                `If you encounter issues, please report at: ${GITHUB_ISSUE_URL}`,
        );
    }

    return MODEL_TYPES.EncoderOnly;
}
