import { AutoTokenizer, Tensor } from "@huggingface/transformers";
import { ResponseConstraint } from "/Users/nico/Documents/Dev/transformers.js/packages/transformers-response-constraint/dist/index.js";

const MODEL = process.argv[2] ?? "onnx-community/gemma-4-E2B-it-ONNX";

const simpleJsonSchemaFormat = {
  type: "json_schema",
  json_schema: {
    "x-guidance": {
      whitespace_flexible: false,
      item_separator: ", ",
      key_separator: ": ",
    },
    type: "object",
    properties: {
      answer: { type: "string", minLength: 1, maxLength: 120 },
    },
    required: ["answer"],
    additionalProperties: false,
  },
};

const complexJsonSchemaFormat = {
  type: "json_schema",
  json_schema: {
    "x-guidance": {
      whitespace_flexible: false,
      item_separator: ", ",
      key_separator: ": ",
    },
    type: "object",
    properties: {
      answer: {
        type: "object",
        properties: {
          text: { type: "string", minLength: 1, maxLength: 120 },
          tone: { enum: ["friendly", "formal", "playful"] },
          language: { enum: ["en", "de", "fr", "es"] },
        },
        required: ["text", "tone", "language"],
        additionalProperties: false,
      },
      alternatives: {
        type: "array",
        minItems: 1,
        maxItems: 3,
        items: { type: "string", minLength: 1, maxLength: 120 },
      },
      metadata: {
        type: "object",
        properties: {
          confidence: { type: "integer", minimum: 0, maximum: 100 },
          safe: { type: "boolean" },
          tags: {
            type: "array",
            minItems: 1,
            maxItems: 3,
            items: { type: "string", minLength: 1, maxLength: 20 },
          },
        },
        required: ["confidence", "safe", "tags"],
        additionalProperties: false,
      },
    },
    required: ["answer", "alternatives", "metadata"],
    additionalProperties: false,
  },
};

const regexFormat = { type: "regex", regex: "(Hello|Hi|Hey)( there)?[!.]" };

const cases = [
  {
    name: "simple JSON",
    format: simpleJsonSchemaFormat,
    output: '{"answer": "Hello there! How can I help you today?"}',
  },
  { name: "regex", format: regexFormat, output: "Hello there!" },
  {
    name: "complex JSON",
    format: complexJsonSchemaFormat,
    output:
      '{"answer": {"text": "Hello there! How can I help you today?", "tone": "friendly", "language": "en"}, "alternatives": ["Hi! What can I do for you?", "Hey, great to see you."], "metadata": {"confidence": 95, "safe": true, "tags": ["greeting", "friendly"]}}',
  },
];

const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const eosId = tokenizer.eos_token_id;
console.log("eos:", tokenizer.eos_token, eosId);

const VOCAB = 262144;

function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * p)];
}

function run(format, tokenIds, label) {
  const t0 = performance.now();
  const constraint = ResponseConstraint.fromResponseFormat(tokenizer, format);
  const setupMs = performance.now() - t0;

  const scores = new Tensor("float32", new Float32Array(VOCAB), [1, VOCAB]);
  const inputIds = [0n];
  const stepMs = [];
  for (const tokenId of tokenIds) {
    scores.data.fill(0);
    const s = performance.now();
    constraint.logits_processor([inputIds], scores);
    if (!Number.isFinite(scores.data[tokenId])) {
      const txt = tokenizer.decode([tokenId]);
      throw new Error(`${label} rejected token ${tokenId} (${JSON.stringify(txt)})`);
    }
    inputIds.push(BigInt(tokenId));
    constraint.logits_processor.onTokensSampled([tokenId], [inputIds]);
    constraint.stopping_criteria([inputIds]);
    stepMs.push(performance.now() - s);
  }
  return { setupMs, stepMs };
}

// EOS-terminated token sequence for each case
for (const c of cases) {
  const ids = tokenizer.encode(c.output, { add_special_tokens: false });
  c.tokenIds = [...ids, Number(eosId)];
}

const results = [];
for (const c of cases) {
  // Run 1: cold (first-ever fromResponseFormat pays tokenizer trie build; caches empty)
  const cold = run(c.format, c.tokenIds, c.name);
  // Run 2: warm-ish (mask caches populated from run 1)
  const warm1 = run(c.format, c.tokenIds, c.name);
  const warm2 = run(c.format, c.tokenIds, c.name);
  results.push({
    name: c.name,
    tokens: c.tokenIds.length,
    "cold setup ms": cold.setupMs.toFixed(1),
    "warm setup ms": warm2.setupMs.toFixed(2),
    "cold sum ms": cold.stepMs.reduce((a, b) => a + b, 0).toFixed(1),
    "cold med": percentile(cold.stepMs, 0.5).toFixed(2),
    "cold p90": percentile(cold.stepMs, 0.9).toFixed(2),
    "cold max": Math.max(...cold.stepMs).toFixed(2),
    "warm sum ms": warm2.stepMs.reduce((a, b) => a + b, 0).toFixed(1),
    "warm med": percentile(warm2.stepMs, 0.5).toFixed(3),
    "warm p90": percentile(warm2.stepMs, 0.9).toFixed(2),
    "warm max": Math.max(...warm2.stepMs).toFixed(2),
  });
}
console.table(results);
