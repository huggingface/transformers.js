import { Tensor } from "@huggingface/transformers";

import { ResponseConstraint } from "../../dist/index.js";

const DEFAULT_EOS = 261;
const DEFAULT_TOKENIZER = {
  tokens: [...Array.from({ length: 256 }, (_, byte) => [byte]), ...Array.from({ length: 6 }, () => [])],
  eos_token_id: DEFAULT_EOS,
  special_token_ids: [256, 257, 258, 259, 260, 261],
};

export async function loadLLGuidance() {
  return runtime;
}

export const loadBundledLLGuidance = loadLLGuidance;

class LLTokenizer {
  constructor(config = {}) {
    this._tokenizer = normalizeTokenizer(config);
    this.eos_token = this._tokenizer.eos_token_id;
    this.eos_tokens = [this.eos_token];
    this.vocab_size = this._tokenizer.tokens.length;
  }

  dispose() {}
  free() {}
}

class LLMatcher {
  static grammar_from_json_schema(schema) {
    return typeof schema === "string" ? JSON.parse(schema) : schema;
  }

  constructor(tokenizer, schema) {
    this._interpreter = createInterpreter({
      tokenizer: unwrapTokenizer(tokenizer),
      response_format: { type: "json_schema", json_schema: schema },
    });
  }

  dispose() {}
  free() {}
}

function createTokenizer(config = {}) {
  return normalizeTokenizer(config);
}

function createInterpreter({ tokenizer, response_format }) {
  const source = unwrapTokenizer(tokenizer);
  const constraint = ResponseConstraint.fromResponseFormat(source, response_format);
  const vocabSize = source.tokens.length;
  let inputIds = [0n];

  return {
    computeMask() {
      const logits = new Tensor("float32", new Float32Array(vocabSize), [1, vocabSize]);
      try {
        constraint.logits_processor([inputIds], logits);
      } catch {
        return { stop: true, reason: "dead_end" };
      }
      const mask = new Uint32Array(Math.ceil(vocabSize / 32));
      for (let token = 0; token < vocabSize; ++token) {
        if (Number.isFinite(logits.data[token])) mask[token >>> 5] |= 1 << (token & 31);
      }
      return { mask, vocabSize };
    },
    computeMaskInto(target) {
      const result = this.computeMask();
      target.fill(0);
      if ("mask" in result) {
        target.set(result.mask);
        return { mask: target, vocabSize };
      }
      return result;
    },
    commitToken(token) {
      inputIds = [...inputIds, BigInt(token)];
      constraint.logits_processor.onTokensSampled([token], [inputIds]);
      return {
        stop: constraint.stopping_criteria([inputIds])[0],
        backtrack: 0,
        ffTokens: [],
      };
    },
    dispose() {},
    free() {},
  };
}

function normalizeTokenizer(config) {
  if (!config.tokens) return DEFAULT_TOKENIZER;
  const eos = config.eosTokenId ?? config.eos_token_id;
  return {
    ...config,
    eos_token_id: eos,
    special_token_ids: config.specialTokenIds ?? config.special_token_ids ?? [eos],
  };
}

function unwrapTokenizer(tokenizer) {
  return tokenizer?._tokenizer ?? tokenizer ?? DEFAULT_TOKENIZER;
}

const runtime = {
  LLTokenizer,
  LLMatcher,
  createTokenizer,
  createInterpreter,
  get_version: () => "llguidance-ts-core",
};
