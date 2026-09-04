import { PreTrainedTokenizer, Tensor } from "@huggingface/transformers";

import { StructuredOutputProcessor } from "../dist/index.js";

const EOS_TOKEN_ID = 256;
const tokenizer = {
  tokens: [...Array.from({ length: EOS_TOKEN_ID }, (_, tokenId) => [tokenId]), []],
  eos_token_id: EOS_TOKEN_ID,
  special_token_ids: [EOS_TOKEN_ID],
};

function logits() {
  return new Tensor("float32", new Float32Array(EOS_TOKEN_ID + 1).fill(1), [1, EOS_TOKEN_ID + 1]);
}

function isAllowed(scores, tokenId) {
  return Number.isFinite(scores.data[tokenId]);
}

const inputIdsByConstraint = new WeakMap();

async function consume(processor, text) {
  const inputIds = inputIdsByConstraint.get(processor) ?? [0n];
  inputIdsByConstraint.set(processor, inputIds);
  for (const tokenId of new TextEncoder().encode(text)) {
    const scores = logits();
    processor([inputIds], scores);
    expect(isAllowed(scores, tokenId)).toBe(true);
    inputIds.push(BigInt(tokenId));
  }
  return inputIds;
}

function schemaAccepts(schema, text) {
  const processor = new StructuredOutputProcessor(tokenizer, {
    type: "json_schema",
    json_schema: schema,
  });
  const inputIds = [0n];
  for (const tokenId of new TextEncoder().encode(text)) {
    const scores = logits();
    try {
      processor([inputIds], scores);
    } catch {
      return false;
    }
    if (!isAllowed(scores, tokenId)) return false;
    inputIds.push(BigInt(tokenId));
  }
  const scores = logits();
  try {
    processor([inputIds], scores);
  } catch {
    return false;
  }
  return isAllowed(scores, EOS_TOKEN_ID);
}

describe("StructuredOutputProcessor", () => {
  it("derives token bytes through the configured decoder", () => {
    const tokenizerJson = {
      version: "1.0",
      truncation: null,
      padding: null,
      added_tokens: [
        {
          id: 1,
          content: "[EOS]",
          single_word: false,
          lstrip: false,
          rstrip: false,
          normalized: false,
          special: true,
        },
      ],
      normalizer: null,
      pre_tokenizer: null,
      post_processor: null,
      decoder: { type: "Replace", pattern: { String: "a" }, content: "b" },
      model: {
        type: "WordPiece",
        unk_token: "[EOS]",
        continuing_subword_prefix: "##",
        max_input_chars_per_word: 100,
        vocab: { a: 0, "[EOS]": 1 },
      },
    };
    const decodedTokenizer = new PreTrainedTokenizer(tokenizerJson, { eos_token: "[EOS]" });
    const processor = new StructuredOutputProcessor(decodedTokenizer, { type: "regex", regex: "b" });
    const scores = new Tensor("float32", new Float32Array(2).fill(1), [1, 2]);

    processor([[0n]], scores);

    expect(decodedTokenizer.decode([0])).toBe("b");
    expect(isAllowed(scores, 0)).toBe(true);
    expect(isAllowed(scores, 1)).toBe(false);
  });

  it("applies a regex mask", async () => {
    const processor = new StructuredOutputProcessor(tokenizer, { type: "regex", regex: "[ac]" });
    const scores = logits();

    processor([[0n]], scores);

    expect(isAllowed(scores, "a".charCodeAt(0))).toBe(true);
    expect(isAllowed(scores, "b".charCodeAt(0))).toBe(false);
    expect(isAllowed(scores, "c".charCodeAt(0))).toBe(true);
    expect(isAllowed(scores, EOS_TOKEN_ID)).toBe(false);
  });

  it("does not process the same sampled token twice", async () => {
    const processor = new StructuredOutputProcessor(tokenizer, { type: "regex", regex: "ab" });
    const inputIds = [0n];
    processor([inputIds], logits());
    inputIds.push(BigInt("a".charCodeAt(0)));

    processor([inputIds], logits());

    const scores = logits();
    processor([inputIds], scores);
    expect(isAllowed(scores, "b".charCodeAt(0))).toBe(true);
  });

  it("discourages repeated non-progressing JSON whitespace", () => {
    const processor = new StructuredOutputProcessor(tokenizer, { type: "json_object" });
    const inputIds = [0n];
    processor([inputIds], logits());

    for (let count = 1; count <= 4; ++count) {
      inputIds.push(10n);

      const scores = logits();
      scores.data[10] = 12;
      scores.data[32] = 12;
      scores.data[13] = -12;
      processor([inputIds], scores);

      if (count < 4) {
        expect(scores.data[10]).toBeCloseTo(12 / 1.2 ** count);
        expect(scores.data[32]).toBeCloseTo(12 / 1.2 ** count);
        expect(scores.data[13]).toBeCloseTo(-12 * 1.2 ** count);
      } else {
        expect(scores.data[10]).toBe(-Infinity);
        expect(scores.data[32]).toBe(-Infinity);
        expect(scores.data[13]).toBe(-Infinity);
      }
      expect(scores.data["{".charCodeAt(0)]).toBe(1);
    }
  });

  it("accepts JSON that satisfies a schema", async () => {
    const processor = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: {
        type: "object",
        properties: { answer: { type: "string", minLength: 1 } },
        required: ["answer"],
        additionalProperties: false,
      },
    });
    const inputIds = await consume(processor, '{"answer":"yes"}');
    const scores = logits();

    processor([inputIds], scores);
    expect(isAllowed(scores, EOS_TOKEN_ID)).toBe(true);
  });

  it("applies schema structure while producing JSON", async () => {
    const constraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: {
        type: "object",
        properties: { answer: { type: "string" } },
        required: ["answer"],
        additionalProperties: false,
      },
    });
    const initial = logits();
    constraint([[0n]], initial);
    expect(isAllowed(initial, "{".charCodeAt(0))).toBe(true);
    expect(isAllowed(initial, "[".charCodeAt(0))).toBe(false);

    const inputIds = await consume(constraint, "{");
    const afterOpen = logits();
    constraint([inputIds], afterOpen);
    expect(isAllowed(afterOpen, "}".charCodeAt(0))).toBe(false);
  });

  it("rejects impossible property and finite scalar prefixes", async () => {
    const constraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: {
        type: "object",
        properties: { answer: { enum: ["yes"] } },
        required: ["answer"],
        additionalProperties: false,
      },
    });
    const propertyInput = await consume(constraint, '{"');
    const propertyScores = logits();
    constraint([propertyInput], propertyScores);
    expect(isAllowed(propertyScores, "a".charCodeAt(0))).toBe(true);
    expect(isAllowed(propertyScores, "z".charCodeAt(0))).toBe(false);

    const valueInput = await consume(constraint, 'answer":"');
    const valueScores = logits();
    constraint([valueInput], valueScores);
    expect(isAllowed(valueScores, "y".charCodeAt(0))).toBe(true);
    expect(isAllowed(valueScores, "x".charCodeAt(0))).toBe(false);

    const unicodeProperty = {
      type: "object",
      properties: { "😀": { type: "string" } },
      required: ["😀"],
      additionalProperties: false,
    };
    expect(schemaAccepts(unicodeProperty, '{"😀":"yes"}')).toBe(true);
    expect(schemaAccepts(unicodeProperty, '{"😁":"yes"}')).toBe(false);

    const escapedProperty = {
      type: "object",
      properties: { confidence: { type: "number" } },
      required: ["confidence"],
      additionalProperties: false,
    };
    expect(schemaAccepts(escapedProperty, '{"c\\u006fnfidence":1}')).toBe(false);
    expect(schemaAccepts(escapedProperty, '{"c\\n\\n\\n":1}')).toBe(false);
    const canonicalKeyConstraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: escapedProperty,
    });
    const canonicalKeyInput = await consume(canonicalKeyConstraint, '{"confidence');
    const canonicalKeyScores = logits();
    canonicalKeyConstraint([canonicalKeyInput], canonicalKeyScores);
    expect(isAllowed(canonicalKeyScores, '"'.charCodeAt(0))).toBe(true);
    expect(isAllowed(canonicalKeyScores, "\\".charCodeAt(0))).toBe(false);
    const completedPropertyInput = await consume(canonicalKeyConstraint, '":1');
    const completedPropertyScores = logits();
    canonicalKeyConstraint([completedPropertyInput], completedPropertyScores);
    expect(isAllowed(completedPropertyScores, "}".charCodeAt(0))).toBe(true);
    expect(isAllowed(completedPropertyScores, ",".charCodeAt(0))).toBe(false);
    expect(schemaAccepts({ type: "object", properties: { "a\nb": true }, required: ["a\nb"], additionalProperties: false }, '{"a\\nb":1}')).toBe(true);

    const languageConstraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: { enum: ["en", "de", "fr", "es"] },
    });
    const languageInput = await consume(languageConstraint, '"');
    const languageScores = logits();
    languageConstraint([languageInput], languageScores);
    expect(isAllowed(languageScores, "e".charCodeAt(0))).toBe(true);
    expect(isAllowed(languageScores, "d".charCodeAt(0))).toBe(true);
    expect(isAllowed(languageScores, "x".charCodeAt(0))).toBe(false);
    expect(isAllowed(languageScores, "\\".charCodeAt(0))).toBe(false);
    expect(schemaAccepts({ const: "\n" }, '"\\n"')).toBe(true);

    const boundedString = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: { type: "string", maxLength: 2 },
    });
    const boundedStringInput = await consume(boundedString, '"ab');
    const boundedStringScores = logits();
    boundedString([boundedStringInput], boundedStringScores);
    expect(isAllowed(boundedStringScores, '"'.charCodeAt(0))).toBe(true);
    expect(isAllowed(boundedStringScores, "c".charCodeAt(0))).toBe(false);
    expect(isAllowed(boundedStringScores, "\\".charCodeAt(0))).toBe(false);
    expect(schemaAccepts({ type: "string", maxLength: 1 }, '"😀"')).toBe(true);
    expect(schemaAccepts({ type: "string", maxLength: 1 }, '"😀x"')).toBe(false);

    const composed = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: {
        oneOf: [{ const: "general" }, { type: "object", properties: { role: { type: "string" } }, required: ["role"], additionalProperties: false }],
      },
    });
    const composedInput = await consume(composed, '{"');
    const composedScores = logits();
    composed([composedInput], composedScores);
    expect(isAllowed(composedScores, "r".charCodeAt(0))).toBe(true);
    expect(isAllowed(composedScores, "z".charCodeAt(0))).toBe(false);
  });

  it("restricts integer fields to reachable canonical syntax", async () => {
    const confidence = {
      type: "object",
      properties: { confidence: { type: "integer", minimum: 0, maximum: 100 } },
      required: ["confidence"],
      additionalProperties: false,
    };

    // Integer fractions may only contain zeros and exponents may not be
    // negative, so "0.9" (which would strand the
    // model in states like "0.9e-" that can never close) is cut off at the "9".
    const constraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: confidence,
    });
    const input = await consume(constraint, '{"confidence":0.');
    const scores = logits();
    constraint([input], scores);
    expect(isAllowed(scores, "0".charCodeAt(0))).toBe(true);
    expect(isAllowed(scores, "9".charCodeAt(0))).toBe(false);

    const exponent = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: confidence,
    });
    const exponentInput = await consume(exponent, '{"confidence":9e');
    const exponentScores = logits();
    exponent([exponentInput], exponentScores);
    expect(isAllowed(exponentScores, "-".charCodeAt(0))).toBe(false);
    // 9e1 = 90 fits [0, 100]; every exponent starting with 3 puts 9e3+ out of range
    expect(isAllowed(exponentScores, "1".charCodeAt(0))).toBe(true);
    expect(isAllowed(exponentScores, "3".charCodeAt(0))).toBe(false);

    expect(schemaAccepts(confidence, '{"confidence":15}')).toBe(true);
    expect(schemaAccepts(confidence, '{"confidence":1.0}')).toBe(true);
    expect(schemaAccepts(confidence, '{"confidence":9e1}')).toBe(true);
    expect(schemaAccepts(confidence, '{"confidence":0.9}')).toBe(false);
    expect(schemaAccepts(confidence, '{"confidence":9e-1}')).toBe(false);

    // Zero padding carries no information, so it is capped: a model stuck on
    // "0" is eventually forced to close instead of streaming digits forever.
    const padded = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: confidence,
    });
    const paddedInput = await consume(padded, '{"confidence":95e000');
    const paddedScores = logits();
    padded([paddedInput], paddedScores);
    expect(isAllowed(paddedScores, "0".charCodeAt(0))).toBe(false);
    expect(isAllowed(paddedScores, "}".charCodeAt(0))).toBe(true);

    // Digits that could never get back into [0, 100] are pruned: after "15",
    // any further digit forces 150+.
    const bounded = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: confidence,
    });
    const boundedInput = await consume(bounded, '{"confidence":15');
    const boundedScores = logits();
    bounded([boundedInput], boundedScores);
    expect(isAllowed(boundedScores, "0".charCodeAt(0))).toBe(false);
    expect(isAllowed(boundedScores, "}".charCodeAt(0))).toBe(true);

    // A first digit that cannot start any in-range integer is masked: 4, 40-49,
    // 400+ all miss [50, 100], while 1 can still reach 100.
    const range = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: { type: "integer", minimum: 50, maximum: 100 },
    });
    const rangeScores = logits();
    range([[0n]], rangeScores);
    expect(isAllowed(rangeScores, "5".charCodeAt(0))).toBe(true);
    expect(isAllowed(rangeScores, "1".charCodeAt(0))).toBe(true);
    expect(isAllowed(rangeScores, "4".charCodeAt(0))).toBe(false);
    expect(isAllowed(rangeScores, "-".charCodeAt(0))).toBe(false);

    const negative = { type: "integer", minimum: -50, maximum: -10 };
    expect(schemaAccepts(negative, "-25")).toBe(true);
    const negativeConstraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: negative,
    });
    const negativeInput = await consume(negativeConstraint, "-");
    const negativeScores = logits();
    negativeConstraint([negativeInput], negativeScores);
    expect(isAllowed(negativeScores, "2".charCodeAt(0))).toBe(true);
    expect(isAllowed(negativeScores, "6".charCodeAt(0))).toBe(false);

    // Integer-valued enums get the same protection.
    expect(schemaAccepts({ enum: [1, 2, 30] }, "30")).toBe(true);
    expect(schemaAccepts({ enum: [1, 2, 30] }, "1.0")).toBe(true);
    expect(schemaAccepts({ enum: [1, 2, 30] }, "1.5")).toBe(false);

    // Plain number fields keep full JSON syntax.
    const ratio = { type: "number", minimum: 0, maximum: 1 };
    expect(schemaAccepts(ratio, "0.9")).toBe(true);
    expect(schemaAccepts(ratio, "9e-1")).toBe(true);
  });

  it("supports composition, conditionals, and local references", async () => {
    expect(schemaAccepts({ not: { type: "string" } }, "42")).toBe(true);
    expect(schemaAccepts({ not: { type: "string" } }, '"no"')).toBe(false);
    expect(schemaAccepts({ allOf: [{ type: "integer", minimum: 2 }, { multipleOf: 2 }] }, "4")).toBe(true);
    expect(schemaAccepts({ oneOf: [{ type: "string" }, { type: "integer" }] }, "2")).toBe(true);

    const conditional = {
      type: "object",
      properties: { kind: { enum: ["text", "count"] }, value: true },
      required: ["kind", "value"],
      if: { properties: { kind: { const: "text" } }, required: ["kind"] },
      then: { properties: { value: { type: "string" } } },
      else: { properties: { value: { type: "integer" } } },
    };
    expect(schemaAccepts(conditional, '{"value":"ok","kind":"text"}')).toBe(true);
    expect(schemaAccepts(conditional, '{"kind":"text","value":2}')).toBe(false);

    const followUp = {
      type: "object",
      properties: { needed: { type: "boolean" }, question: { type: ["string", "null"] } },
      required: ["needed", "question"],
      additionalProperties: false,
      allOf: [
        {
          if: { properties: { needed: { const: true } }, required: ["needed"] },
          then: { properties: { question: { type: "string", minLength: 1 } } },
          else: { properties: { question: { type: "null" } } },
        },
      ],
    };
    const followUpConstraint = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: followUp,
    });
    const followUpInput = await consume(followUpConstraint, '{"needed":true,"question":');
    const followUpScores = logits();
    followUpConstraint([followUpInput], followUpScores);
    expect(isAllowed(followUpScores, '"'.charCodeAt(0))).toBe(true);
    expect(isAllowed(followUpScores, "n".charCodeAt(0))).toBe(false);
    expect(schemaAccepts(followUp, '{"question":null,"needed":true}')).toBe(false);

    const referenced = {
      $defs: { answer: { type: "integer", minimum: 2 } },
      $ref: "#/$defs/answer",
    };
    expect(schemaAccepts(referenced, "3")).toBe(true);
    expect(schemaAccepts(referenced, "1")).toBe(false);

    const objectUnion = {
      anyOf: [
        { type: "object", properties: { a: { type: "string" }, b: { type: "integer" } }, required: ["a"], additionalProperties: false },
        { type: "object", properties: { a: { type: "string" }, c: { type: "number" } }, required: ["a"], additionalProperties: false },
      ],
    };
    expect(schemaAccepts(objectUnion, '{"a":"x","b":2}')).toBe(true);
    expect(schemaAccepts(objectUnion, '{"a":"x","b":2,"c":3}')).toBe(false);
  });

  it("supports deep equality and structural assertions", () => {
    expect(schemaAccepts({ const: { name: "John", values: [1] } }, '{"values":[1.0],"name":"John"}')).toBe(true);
    expect(schemaAccepts({ const: "😀" }, '"\\ud83d\\ude00"')).toBe(true);
    expect(schemaAccepts({ const: "😀" }, '"\\ud83dx"')).toBe(false);
    expect(schemaAccepts({ type: "array", uniqueItems: true }, '[{"a":[1]},{"a":[1.0]}]')).toBe(false);
    expect(schemaAccepts({ type: "array", contains: { type: "integer", minimum: 2 }, minContains: 2, maxContains: 2 }, '["x",2,3]')).toBe(true);
    expect(schemaAccepts({ type: "array", contains: { const: 1 } }, "[0]")).toBe(false);
  });

  it("supports property patterns and dependencies", () => {
    const schema = {
      type: "object",
      properties: { code: { type: "integer" }, card: { type: "string" }, billing: { type: "string" } },
      patternProperties: { "^code$": { minimum: 1 }, "^x-": { type: "string" } },
      dependentRequired: { card: ["billing"] },
      additionalProperties: false,
    };
    expect(schemaAccepts(schema, '{"x-note":"ok","code":2,"billing":"x","card":"1"}')).toBe(true);
    expect(schemaAccepts(schema, '{"code":0}')).toBe(false);
    expect(schemaAccepts(schema, '{"card":"1"}')).toBe(false);
  });

  it("supports recursive references and draft-07 compatibility", () => {
    const linkedList = {
      $defs: {
        node: {
          type: "object",
          properties: {
            value: { type: "string" },
            next: { anyOf: [{ $ref: "#/$defs/node" }, { type: "null" }] },
          },
          required: ["value", "next"],
          additionalProperties: false,
        },
      },
      $ref: "#/$defs/node",
    };
    expect(schemaAccepts(linkedList, '{"value":"a","next":{"value":"b","next":null}}')).toBe(true);
    expect(schemaAccepts(linkedList, '{"value":"a","next":2}')).toBe(false);

    const tuple = {
      type: "array",
      items: [{ type: "string" }, { type: "integer" }],
      additionalItems: false,
    };
    expect(schemaAccepts(tuple, '["x",1]')).toBe(true);
    expect(schemaAccepts(tuple, '["x",1,true]')).toBe(false);
  });

  it("supports x-guidance separators", () => {
    const guided = {
      type: "object",
      properties: { a: { type: "integer" }, b: { type: "integer" } },
      required: ["a", "b"],
      additionalProperties: false,
      "x-guidance": { item_separator: "-", key_separator: "_ ", whitespace_flexible: false },
    };
    expect(schemaAccepts(guided, '{"a"_ 1-"b"_ 2}')).toBe(true);
    expect(schemaAccepts(guided, '{"a":1,"b":2}')).toBe(false);
  });

  it("rejects malformed and unsupported schema shapes", () => {
    for (const schema of [{ not: [] }, { patternProperties: { "[": true } }, { dependentRequired: { a: ["b", "b"] } }, { minContains: -1 }, { uniqueItems: "yes" }, { unevaluatedProperties: false }]) {
      expect(() => new StructuredOutputProcessor(tokenizer, { type: "json_schema", json_schema: schema })).toThrow();
    }
  });

  it("rejects string assertions that cannot be enforced incrementally", () => {
    for (const schema of [
      { type: "string", pattern: "^yes$" },
      { type: "object", properties: { value: { type: "string", format: "date" } } },
    ]) {
      expect(() => new StructuredOutputProcessor(tokenizer, { type: "json_schema", json_schema: schema })).toThrow("cannot be enforced incrementally");
    }
    expect(schemaAccepts({ type: "string", format: "vendor-format" }, '"anything"')).toBe(true);
  });

  it("supports unconstrained JSON objects", async () => {
    const constraint = new StructuredOutputProcessor(tokenizer, { type: "json_object" });
    const inputIds = await consume(constraint, '{"nested":{"enabled":true},"count":2}');
    const scores = logits();

    constraint([inputIds], scores);

    expect(isAllowed(scores, EOS_TOKEN_ID)).toBe(true);
  });

  it("reuses cached masks without sharing generation state", async () => {
    const schema = {
      type: "object",
      properties: { answer: { enum: ["yes", "no"] } },
      required: ["answer"],
      additionalProperties: false,
    };
    const first = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: schema,
    });
    const second = new StructuredOutputProcessor(tokenizer, {
      type: "json_schema",
      json_schema: schema,
    });

    const firstIds = await consume(first, '{"answer":"yes"}');
    const secondIds = await consume(second, '{"answer":"no"}');
    const firstScores = logits();
    const secondScores = logits();
    first([firstIds], firstScores);
    second([secondIds], secondScores);

    expect(isAllowed(firstScores, EOS_TOKEN_ID)).toBe(true);
    expect(isAllowed(secondScores, EOS_TOKEN_ID)).toBe(true);
  });

  it("rejects batched generation", async () => {
    const constraint = new StructuredOutputProcessor(tokenizer, { type: "json_object" });

    expect(() => constraint([[0n], [0n]], new Tensor("float32", new Float32Array((EOS_TOKEN_ID + 1) * 2), [2, EOS_TOKEN_ID + 1]))).toThrow("currently supports batch size 1");
  });

  it("rejects a sampled token outside the constraint", async () => {
    const constraint = new StructuredOutputProcessor(tokenizer, { type: "regex", regex: "a" });
    constraint([[0n]], logits());

    expect(() => constraint([[0n, 98n]], logits())).toThrow("does not satisfy the constraint");
  });

  it("can be passed directly as a logits processor list", () => {
    const constraint = new StructuredOutputProcessor(tokenizer, { type: "regex", regex: "a" });

    expect([...constraint]).toHaveLength(1);
    expect("stopping_criteria" in constraint).toBe(false);
  });
});
