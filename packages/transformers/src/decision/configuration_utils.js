/**
 * @file Settings of decision models (experimental, ported from `transformers.decision`).
 *
 * @module decision/configuration_utils
 */

import { getModelJSON, getModelText } from '../utils/hub.js';
import { prepare_inputs } from './template_utils.js';

const DECISION_CONFIG_NAME = 'decision.json';
const DECISION_TEMPLATE_NAME = 'decision_template.jinja';

/**
 * Settings of a decision model, saved as `decision.json` (and its chat template as `decision_template.jinja`).
 */
export class DecisionConfig {
    /**
     * Single-token labels of the options per question type, for heads that read them from the logits. Defaults to
     * `Yes`/`No` for `noul` and `A`, `B`, `C`, ... otherwise.
     * @type {Record<string, string[]>}
     */
    labels = {};

    /**
     * Temperature per question type, applied to the option scores before the softmax. Defaults to 1.
     * @type {Record<string, number>}
     */
    temperature = {};

    /**
     * Jinja template rendering `state` with `question` (one prompt per question) or `questions` (one prompt for all
     * of them). Defaults to a generic prompt in the tokenizer's chat template.
     * @type {string|null}
     */
    chat_template = null;

    /**
     * Order in which the template receives the two options of a `noul` question.
     * @type {string[]}
     */
    noul_options = ['true', 'false'];

    /**
     * Whether control tokens in the request are encoded as text. Templates that escape them themselves disable it.
     * @type {boolean}
     */
    sanitize_control_tokens = true;

    /**
     * Subfolder of a LoRA adapter. Not supported in Transformers.js: merge the adapter into the exported weights.
     * @type {string|null}
     */
    adapter = null;

    /**
     * Create a new `DecisionConfig`. Settings this version doesn't know are ignored.
     * @param {Partial<DecisionConfig>} [config] The settings, e.g. the contents of a `decision.json`.
     */
    constructor(config = {}) {
        for (const key of Object.keys(this)) {
            if (key in config && config[key] !== undefined) this[key] = config[key];
        }
        const sorted = [...this.noul_options].sort();
        if (sorted.length !== 2 || sorted[0] !== 'false' || sorted[1] !== 'true') {
            throw new Error(`noul_options must order "true" and "false", got ${JSON.stringify(this.noul_options)}.`);
        }
        for (const [kind, value] of Object.entries(this.temperature)) {
            if (typeof value !== 'number' || !(value > 0)) {
                throw new Error(
                    `temperature.${kind} must be a strictly positive number, got ${JSON.stringify(value)}.`,
                );
            }
        }
    }

    /**
     * The temperature applied to the option scores of a question of type `question_type`.
     * @param {string} question_type
     * @returns {number}
     */
    get_temperature(question_type) {
        return this.temperature[question_type] ?? 1.0;
    }

    /**
     * The tokenized prompts of `request`, see `prepare_inputs`.
     * @param {import('./template_utils.js').DecisionRequest|string} request
     * @param {import('../tokenization_utils.js').PreTrainedTokenizer} tokenizer
     * @returns {import('./template_utils.js').DecisionPrompt[]}
     */
    prepare_inputs(request, tokenizer) {
        return prepare_inputs(this, request, tokenizer);
    }

    /**
     * Loads the decision config of a model repo: its `decision.json` and `decision_template.jinja`, or the default
     * settings for a repo without them.
     * @param {string} pretrained_model_name_or_path The model id or path.
     * @param {import('../utils/hub.js').PretrainedOptions} [options] Loading options.
     * @returns {Promise<DecisionConfig>}
     */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        if (!pretrained_model_name_or_path) return new DecisionConfig();
        const [config, chat_template] = await Promise.all([
            getModelJSON(pretrained_model_name_or_path, DECISION_CONFIG_NAME, false, options),
            getModelText(pretrained_model_name_or_path, DECISION_TEMPLATE_NAME, false, options),
        ]);
        return new DecisionConfig({ ...config, ...(chat_template !== null ? { chat_template } : {}) });
    }
}
