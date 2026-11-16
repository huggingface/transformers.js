/**
 * @file Prompts of decision requests (experimental, ported from `transformers.decision`).
 *
 * A request `{ state, questions: { name: question } }` is rendered into one prompt per question, or one for all of
 * them when the template reads `questions`. A decision template places `<|mark|>` where each answer is read (the
 * token before it) and `<|split|>` to split the encoding; both are dropped from the input ids. Models without a
 * decision template use `DECISION_PROMPT` as the user message of their chat template, and are read after the
 * generation prompt.
 *
 * @module decision/template_utils
 */

import { Template } from '@huggingface/jinja';

import { max } from '../utils/maths.js';

const MARK = '<|mark|>';
const SPLIT = '<|split|>';

// Private-use characters that wrap the control tokens of user strings, so that they are encoded as text
const CONTROL_TOKEN_START = '\u{F0000}';
const CONTROL_TOKEN_END = '\u{F0001}';

/**
 * The default prompt, the one of the `/v1/systemone` endpoint of `transformers serve`. It renders the same text as the
 * Python one, without `select`, `list` on strings or list equality, which `@huggingface/jinja` doesn't have.
 */
export const DECISION_PROMPT = `
{%- macro text(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{%- set instructions = text(question.instructions) -%}
{%- set ns = namespace(lines=[instructions] if instructions else []) -%}
{%- for option in question.options -%}
    {%- set label, answer = question.labels[loop.index0], text(question.descriptions[loop.index0]) -%}
    {%- if question.type == "noul" -%}
        {%- set answer = answer or (option if not (question.labels[0] == "Yes" and question.labels[1] == "No") else "") -%}
        {%- set ns.lines = ns.lines + ([label ~ ": " ~ answer] if answer else []) -%}
    {%- else -%}
        {%- set answer = (option ~ ": " ~ answer if answer else option) if question.type == "choice" else answer -%}
        {%- set ns.lines = ns.lines + [label ~ ". " ~ answer] -%}
    {%- endif -%}
{%- endfor -%}
{{- text(state) ~ "\\n\\n" ~ ns.lines|join("\\n") ~ "\\n" -}}
{%- if question.type == "noul" -%}
    {{- "Answer " ~ question.labels[0] ~ " or " ~ question.labels[1] ~ " only." -}}
{%- else -%}
    {%- set kind = "letter" if question.labels|join("") == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: question.labels|length] else "label" -%}
    {{- "Answer with the " ~ kind ~ " of the " ~ ("option" if question.type == "choice" else "level") -}}
    {{- " that fits best (" ~ question.labels[0] ~ " to " ~ question.labels[-1] ~ ") only." -}}
{%- endif -%}
`;

const QUESTION_TYPES = ['noul', 'choice', 'score'];

const QUESTION_EXAMPLES = {
    noul: '{"type": "noul", "instructions": "Is this message spam?", "criteria": {"true": "Unsolicited ads", "false": "A real conversation"}}',
    choice: '{"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": "A polite message", "angry": null}}',
    score: '{"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "This week", "Today"]}',
};

/** @param {any} x */
const isObject = (x) => typeof x === 'object' && x !== null && !Array.isArray(x);

/** @param {any} x */
const isContent = (x) => typeof x === 'string' || (typeof x === 'object' && x !== null);

/** @param {string} s */
const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * @typedef {Object} QuestionRequest A question of a decision request.
 * @property {'noul'|'choice'|'score'} type The type of the question.
 * @property {string|Object|Array|null} [instructions] What is asked.
 * @property {Object|Array} [criteria] The options: for `choice`, an object mapping each option to its description (or
 * `null`); for `score`, the levels, lowest first; for `noul`, optional descriptions of `true` and `false`.
 *
 * @typedef {Object} DecisionRequest A decision request, in the System One format.
 * @property {any} state What the questions are about.
 * @property {Record<string, QuestionRequest>} questions The questions, by name.
 */

/**
 * The fields check of the strict question dataclasses of the Python port.
 * @param {string} kind
 * @param {Object} question
 * @returns {string|undefined} The error, if any.
 */
function validate(kind, { type, instructions = null, criteria = null, ...unexpected }) {
    const [key] = Object.keys(unexpected);
    if (key) return `unexpected keyword argument '${key}'`;
    if (instructions !== null && !isContent(instructions)) {
        return `\`instructions\` must be a string, an object, an array or null, got ${JSON.stringify(instructions)}`;
    }
    if (criteria === null) return kind === 'noul' ? undefined : 'missing required argument `criteria`';
    const valid =
        kind === 'score'
            ? Array.isArray(criteria) && criteria.every(isContent)
            : isObject(criteria) &&
              Object.entries(criteria).every(
                  ([option, value]) =>
                      (kind === 'choice' || option === 'true' || option === 'false') &&
                      (value === null || isContent(value)),
              );
    if (!valid) return `invalid \`criteria\` ${JSON.stringify(criteria)}`;
}

/**
 * @typedef {Object} NoulAnswer
 * @property {'noul'} type
 * @property {number} noul The probability of `true`.
 *
 * @typedef {Object} ChoiceAnswer
 * @property {'choice'} type
 * @property {string} choice The most probable option.
 * @property {number} confidence How far the top probability is above the uniform one, from 0 to 1.
 * @property {Record<string, number>} probabilities The probability of each option.
 *
 * @typedef {Object} ScoreAnswer
 * @property {'score'} type
 * @property {number} score The expected level (from 0, the lowest level).
 * @property {Record<string, any>} legend The levels, by index.
 * @property {number} confidence How concentrated the probabilities are around the top level, from 0 to 1.
 * @property {Record<string, number>} probabilities The probability of each level, by index.
 *
 * @typedef {NoulAnswer|ChoiceAnswer|ScoreAnswer} DecisionAnswer
 */

/**
 * One System One question, with its options (choice keys, score levels or true/false) in template order.
 */
export class Question {
    /**
     * @param {string} name
     * @param {'noul'|'choice'|'score'} type
     * @param {any} instructions
     * @param {string[]} options
     * @param {string[]} labels
     * @param {any[]} descriptions
     */
    constructor(name, type, instructions, options, labels, descriptions) {
        this.name = name;
        this.type = type;
        this.instructions = instructions;
        this.options = options;
        this.labels = labels;
        this.descriptions = descriptions;
    }

    /**
     * Validates a question of a request.
     * @param {string} name The name of the question.
     * @param {any} question The question.
     * @param {import('./configuration_utils.js').DecisionConfig} config The decision config.
     * @returns {Question}
     */
    static from_request(name, question, config) {
        const kind = isObject(question) ? question.type : undefined;
        if (!QUESTION_TYPES.includes(kind)) {
            const got = isObject(question) ? kind : question;
            throw new Error(
                `Question "${name}": \`type\` must be one of ${QUESTION_TYPES.join(', ')}, got ${JSON.stringify(got)}.`,
            );
        }
        const error = validate(kind, question);
        if (error) {
            throw new Error(`Question "${name}": ${error} (a ${kind} question looks like ${QUESTION_EXAMPLES[kind]})`);
        }

        let criteria = question.criteria ?? null;
        if (kind !== 'noul' && Object.keys(criteria).length === 0) {
            throw new Error(`Question "${name}": a ${kind} question needs \`criteria\` with its options.`);
        }
        if (kind === 'noul') {
            criteria = Object.fromEntries(config.noul_options.map((option) => [option, criteria?.[option] ?? null]));
        } else if (kind === 'score') {
            criteria = Object.fromEntries(criteria.map((level, index) => [String(index), level]));
        }

        const labels = config.labels[kind]?.length
            ? config.labels[kind]
            : kind === 'noul'
              ? ['Yes', 'No']
              : [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];
        const options = Object.keys(criteria);
        if (labels.length < options.length) {
            throw new Error(
                `Question "${name}": this model reads at most ${labels.length} options for a ${kind} question, got ` +
                    `${options.length} (\`labels.${kind}\` in its decision config).`,
            );
        }
        const instructions = question.instructions ?? null;
        return new Question(
            name,
            kind,
            instructions,
            options,
            labels.slice(0, options.length),
            Object.values(criteria),
        );
    }

    /**
     * The System One answer, from the probabilities of the options.
     * @param {number[]} probabilities The probability of each option, in order.
     * @returns {DecisionAnswer}
     */
    answer(probabilities) {
        const by_option = Object.fromEntries(this.options.map((option, i) => [option, probabilities[i]]));
        const n_options = probabilities.length;
        if (this.type === 'noul') {
            return { type: 'noul', noul: by_option['true'] };
        }

        const [top, best] = max(probabilities);
        if (this.type === 'choice') {
            const confidence = n_options === 1 ? 1.0 : Math.max(0.0, (top - 1 / n_options) / (1 - 1 / n_options));
            return { type: 'choice', choice: this.options[best], confidence, probabilities: by_option };
        }

        let distance = 0;
        let uniform_distance = 0;
        let score = 0;
        probabilities.forEach((probability, level) => {
            distance += probability * Math.abs(level - best);
            uniform_distance += Math.abs(level - (n_options - 1) / 2);
            score += level * probability;
        });
        const confidence = n_options === 1 ? 1.0 : Math.max(0.0, 1 - distance / (uniform_distance / n_options));
        return {
            type: 'score',
            score,
            legend: Object.fromEntries(this.options.map((option, i) => [option, this.descriptions[i]])),
            confidence,
            probabilities: by_option,
        };
    }
}

/**
 * The ids of the `labels` of a question, which must be distinct single known tokens: without a decision head, the
 * score of each option is the logit of its label token, at the token before the mark.
 * @param {import('../tokenization_utils.js').PreTrainedTokenizer} tokenizer
 * @param {string[]} labels
 * @returns {number[]}
 */
export function label_token_ids(tokenizer, labels) {
    const ids = labels.map((label) => tokenizer.encode(label, { add_special_tokens: false }));
    const label_ids = ids
        .filter((token_ids) => token_ids.length === 1 && token_ids[0] !== tokenizer.unk_token_id)
        .map((token_ids) => token_ids[0]);
    if (new Set(label_ids).size !== ids.length) {
        throw new Error(`The labels ${JSON.stringify(labels)} are not distinct single tokens for this model.`);
    }
    return label_ids;
}

/**
 * Validates the `questions` of a System One request and puts them in template order, throwing one error that lists
 * every invalid question, each with an example of a valid one.
 * @param {any} questions
 * @param {import('./configuration_utils.js').DecisionConfig} config
 * @returns {Question[]}
 */
export function parse_questions(questions, config) {
    if (!isObject(questions) || Object.keys(questions).length === 0) {
        throw new Error(
            `\`questions\` maps names to questions, with at least one question; got ${JSON.stringify(questions)}.`,
        );
    }
    const parsed = [];
    const errors = [];
    for (const [name, question] of Object.entries(questions)) {
        try {
            parsed.push(Question.from_request(name, question, config));
        } catch (error) {
            errors.push(error.message);
        }
    }
    if (errors.length) {
        throw new Error(errors.join('\n'));
    }
    return parsed;
}

/**
 * @param {any} node
 * @param {(node: any) => void} visit
 */
function walk(node, visit) {
    if (Array.isArray(node)) return node.forEach((child) => walk(child, visit));
    if (!isObject(node)) return;
    visit(node);
    for (const [key, value] of Object.entries(node)) {
        if (node.type === 'MemberExpression' && key === 'property' && !node.computed) continue;
        if (node.type === 'KeywordArgumentExpression' && key === 'key') continue;
        walk(value, visit);
    }
}

/**
 * The variables a template reads without setting them, and its literal strings joined with null characters (so that
 * a substring search can't match across two of them), like `jinja2.meta` in the Python port.
 * @param {string} chat_template
 */
function templateInfo(chat_template) {
    const used = new Set();
    const declared = new Set();
    const literals = new Set();
    /** @param {any} target */
    const declare = (target) => walk(target, (node) => node.type === 'Identifier' && declared.add(node.value));
    walk(new Template(chat_template).parsed, (node) => {
        if (node.type === 'Identifier') used.add(node.value);
        else if (node.type === 'StringLiteral') literals.add(node.value);
        else if (node.type === 'Set') declare(node.assignee);
        else if (node.type === 'For') declare(node.loopvar);
        else if (node.type === 'Macro') declare([node.name, node.args]);
    });
    return {
        variables: new Set([...used].filter((name) => !declared.has(name))),
        literal_text: [...literals].join('\0'),
    };
}

/**
 * A regex that matches any of `tokens`, longest first.
 * @param {string[]} tokens
 */
const tokensPattern = (tokens) =>
    [...new Set(tokens)]
        .sort((a, b) => b.length - a.length)
        .map(escapeRegExp)
        .join('|') || '(?!)';

/**
 * Wraps the control tokens in user strings with private-use markers, so that they are encoded as text. Object keys
 * aren't marked.
 * @param {any} obj
 * @param {RegExp} control_tokens_re
 * @returns {any}
 */
function mark_control_tokens(obj, control_tokens_re) {
    if (typeof obj === 'string') {
        // Drop any markers already in the string first, so they can't be used to forge a marked token
        obj = obj.replaceAll(CONTROL_TOKEN_START, '').replaceAll(CONTROL_TOKEN_END, '');
        return obj.replace(control_tokens_re, (token) => CONTROL_TOKEN_START + token + CONTROL_TOKEN_END);
    }
    if (Array.isArray(obj)) return obj.map((value) => mark_control_tokens(value, control_tokens_re));
    if (isObject(obj)) {
        return Object.fromEntries(
            Object.entries(obj).map(([key, value]) => [key, mark_control_tokens(value, control_tokens_re)]),
        );
    }
    return obj;
}

/**
 * Encodes `text` as plain text, without matching any added tokens.
 * @param {import('../tokenization_utils.js').PreTrainedTokenizer} tokenizer
 * @param {string} text
 * @returns {number[]}
 */
function encodeWithoutAddedTokens(tokenizer, text) {
    const { normalizer, pre_tokenizer, model } = tokenizer._tokenizer;
    if (normalizer !== null) text = normalizer(text);
    if (text.length === 0) return [];
    const tokens = model(pre_tokenizer !== null ? pre_tokenizer(text, { section_index: 0 }) : [text]);
    return tokens.map((token) => model.tokens_to_ids.get(token) ?? model.unk_token_id);
}

/**
 * Encodes a piece from the middle of a chat, ignoring added tokens if `as_text` is set. It is encoded after a newline,
 * so SentencePiece doesn't add the prefix space it adds at the start of a string, then the newline is removed.
 * @param {import('../tokenization_utils.js').PreTrainedTokenizer} tokenizer
 * @param {string} piece
 * @param {boolean} as_text
 * @param {number[]} newline_ids
 * @returns {number[]}
 */
function encodeChatPiece(tokenizer, piece, as_text, newline_ids) {
    /** @param {string} x */
    const encode = (x) =>
        as_text ? encodeWithoutAddedTokens(tokenizer, x) : tokenizer.encode(x, { add_special_tokens: false });
    const piece_ids = encode('\n' + piece);
    if (newline_ids.every((id, i) => piece_ids[i] === id)) {
        return piece_ids.slice(newline_ids.length);
    }
    return encode(piece);
}

/**
 * Encodes a rendered prompt without its sentinels, and returns the number of tokens before each `<|mark|>` (or all of
 * them, for a prompt without marks). With `sanitize`, the control tokens marked by `mark_control_tokens` are encoded
 * as text.
 * @param {import('../tokenization_utils.js').PreTrainedTokenizer} tokenizer
 * @param {string} prompt
 * @param {boolean} sanitize
 * @returns {{ input_ids: number[], marks: number[] }}
 */
function encodePrompt(tokenizer, prompt, sanitize) {
    const newline_ids = tokenizer.encode('\n', { add_special_tokens: false });
    // Splitting on the markers gives [text, marked token, text, ..., text], with undefined for a lone marker
    let pieces = [prompt];
    if (sanitize) {
        const added_tokens = [...tokenizer._tokenizer.get_added_tokens_decoder().values()].map((t) => t.content);
        const tokens = tokensPattern([...added_tokens, MARK, SPLIT]);
        const start = CONTROL_TOKEN_START;
        const end = CONTROL_TOKEN_END;
        pieces = prompt.split(new RegExp(`${start}(${tokens})${end}|[${start}${end}]`, 'u'));
    }
    /** @type {number[]} */
    const input_ids = [];
    /** @type {number[]} */
    const marks = [];
    pieces.forEach((piece, i) => {
        if (!piece) return;
        if (i % 2 === 1) {
            const piece_ids = encodeChatPiece(tokenizer, piece, true, newline_ids);
            if (piece_ids.includes(tokenizer._tokenizer.token_to_id(piece))) {
                throw new Error(`Can't sanitize ${JSON.stringify(piece)}: the tokenizer produces it from plain text`);
            }
            input_ids.push(...piece_ids);
            return;
        }
        // The sentinels split the text, as the added tokens they are in the Python port
        piece.split(/(<\|mark\|>|<\|split\|>)/).forEach((part, j) => {
            if (part === MARK) {
                marks.push(input_ids.length);
            } else if (part && part !== SPLIT) {
                input_ids.push(
                    ...(i === 0 && j === 0
                        ? tokenizer.encode(part, { add_special_tokens: false })
                        : encodeChatPiece(tokenizer, part, false, newline_ids)),
                );
            }
        });
    });
    return { input_ids, marks: marks.length ? marks : [input_ids.length] };
}

/**
 * @typedef {Object} DecisionPrompt A tokenized prompt of a decision request.
 * @property {number[]} input_ids The ids of the prompt, without the sentinels.
 * @property {number[]} marks The number of tokens before each mark: where the answers are read.
 * @property {Question[]} questions The questions the prompt asks.
 */

/**
 * Tokenizes a decision `request` (`{ state, questions }`, or its JSON) into one prompt per question, or a single
 * prompt when the template reads `questions`.
 * @param {import('./configuration_utils.js').DecisionConfig} config The decision config.
 * @param {DecisionRequest|string} request The request.
 * @param {import('../tokenization_utils.js').PreTrainedTokenizer} tokenizer The tokenizer of the model.
 * @returns {DecisionPrompt[]}
 */
export function prepare_inputs(config, request, tokenizer) {
    // A string is parsed, so that `\u` escapes can't hide control tokens
    /** @type {any} */
    let parsed = typeof request === 'string' ? JSON.parse(request) : request;
    if (!isObject(parsed) || !('questions' in parsed) || !('state' in parsed)) {
        throw new Error('A request is {"state": ..., "questions": {name: question, ...}}, as an object or its JSON.');
    }
    if (parsed.assets) {
        throw new Error('Image assets are not supported yet by the decision pipeline.');
    }
    if (!config.sanitize_control_tokens) {
        // The template doesn't escape user text: strip the sentinels from it
        parsed = JSON.parse(JSON.stringify(parsed).replaceAll(MARK, '').replaceAll(SPLIT, ''));
    }
    const questions = parse_questions(parsed.questions, config);
    const state = parsed.state;

    const template_input_names = templateInfo(config.chat_template ?? DECISION_PROMPT).variables;
    const groups = template_input_names.has('questions') ? [questions] : questions.map((question) => [question]);

    // The control tokens of the chat template: the special tokens, and the (non-whitespace) added tokens it contains
    const chat_template = config.chat_template ?? tokenizer.get_chat_template();
    const { literal_text } = templateInfo(chat_template);
    const control_tokens = [...tokenizer._tokenizer.get_added_tokens_decoder().values()]
        .filter(({ content, special }) => special || (literal_text.includes(content) && content.trim() !== ''))
        .map(({ content }) => content);
    const control_tokens_re = new RegExp(tokensPattern([...control_tokens, MARK, SPLIT]), 'g');
    /** @param {any} x */
    const mark = (x) => (config.sanitize_control_tokens ? mark_control_tokens(x, control_tokens_re) : x);

    return groups.map((group) => {
        /** @type {Record<string, any>} */
        let template_inputs = {};
        let content = '';
        if (config.chat_template === null) {
            content = new Template(DECISION_PROMPT).render({ state, question: group[0] });
        } else {
            const all_inputs = { state, question: group[0], questions: group };
            template_inputs = Object.fromEntries(
                Object.entries(all_inputs).filter(([name]) => template_input_names.has(name)),
            );
        }
        const rendered = /** @type {string} */ (
            /** @type {unknown} */ (
                tokenizer.apply_chat_template(mark([{ role: 'user', content }]), {
                    chat_template,
                    add_generation_prompt: true,
                    tokenize: false,
                    enable_thinking: false,
                    ...mark(template_inputs),
                })
            )
        );
        return { ...encodePrompt(tokenizer, rendered, config.sanitize_control_tokens), questions: group };
    });
}
