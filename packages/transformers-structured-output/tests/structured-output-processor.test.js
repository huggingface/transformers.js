import { LogitsProcessorList, PreTrainedTokenizer, Tensor, TextGenerationPipeline } from "@huggingface/transformers";

import { StructuredOutputProcessor } from "../dist/index.js";

const EOS_TOKEN_ID = 256;
const tokenizer = {
  tokens: [...Array.from({ length: EOS_TOKEN_ID }, (_, tokenId) => [tokenId]), []],
  eos_token_id: EOS_TOKEN_ID,
  special_token_ids: [EOS_TOKEN_ID],
};

const thinkingTokenizer = {
  ...tokenizer,
  response_template: {
    start_anchor: "<assistant>",
    fields: {
      thinking: { open: "<think>", close: "</think>", content: "text" },
      content: { close: "<eos>", content: "text" },
    },
  },
};

const toolTokenizer = {
  ...thinkingTokenizer,
  response_template: {
    ...thinkingTokenizer.response_template,
    fields: {
      ...thinkingTokenizer.response_template.fields,
      tool_calls: {
        open: "<tool_call>",
        close: "</tool_call>",
        repeats: true,
        content: "json",
      },
    },
  },
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

  it("allows optional thinking before applying a JSON schema", async () => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, {
      type: "json_schema",
      json_schema: {
        type: "object",
        properties: { answer: { type: "string" } },
        required: ["answer"],
        additionalProperties: false,
      },
    });
    const initial = logits();

    processor([[0n]], initial);

    expect(isAllowed(initial, "<".charCodeAt(0))).toBe(true);
    expect(isAllowed(initial, "{".charCodeAt(0))).toBe(true);
    expect(isAllowed(initial, "x".charCodeAt(0))).toBe(false);

    const inputIds = await consume(processor, '<think>free-form reasoning</think>{"answer":"yes"}');
    const final = logits();
    processor([inputIds], final);
    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it("applies constraints immediately when thinking is skipped", async () => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "answer: \\d+" });
    const inputIds = await consume(processor, "answer: 42");
    const final = logits();

    processor([inputIds], final);

    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it("allows a tool-call-only generation before a constrained final response", async () => {
    const responseFormat = { type: "regex", regex: "The answer is \\d+\\." };
    const first = new StructuredOutputProcessor(toolTokenizer, responseFormat);
    const firstIds = await consume(first, '<think>I should calculate this.</think>\n\n<tool_call>{"name":"add","arguments":{"a":4,"b":5}}</tool_call>');
    const afterToolCall = logits();

    first([firstIds], afterToolCall);

    expect(isAllowed(afterToolCall, EOS_TOKEN_ID)).toBe(true);

    const second = new StructuredOutputProcessor(toolTokenizer, responseFormat);
    const secondIds = await consume(second, "<think>The tool returned 9.</think>The answer is 9.");
    const afterContent = logits();
    second([secondIds], afterContent);

    expect(isAllowed(afterContent, EOS_TOKEN_ID)).toBe(true);
  });

  it.each(["(?P<name>\\w+)", "(?<name>\\w+)"])("accepts Gemma tool openers with named captures: %s", async (capture) => {
    const gemmaTokenizer = {
      ...tokenizer,
      response_template: {
        start_anchor: ["<|turn>model\n", "<tool_response|>"],
        fields: {
          thinking: { open: "<|channel>thought\n", close: "<channel|>", content: "text" },
          tool_calls: {
            open_pattern: `<\\|tool_call>call:${capture}`,
            close: "<tool_call|>",
            repeats: true,
            content: "json",
          },
          content: { close: ["<turn|>", "<|tool_response>", "<eos>"], content: "text" },
        },
      },
    };
    const responseFormat = { type: "regex", regex: "The answer is \\d+\\." };
    const first = new StructuredOutputProcessor(gemmaTokenizer, responseFormat);
    const firstIds = await consume(first, "<|channel>thought\nUse add.<channel|><|tool_call>call:add{a:4,b:5}<tool_call|>");
    const afterTool = logits();
    first([firstIds], afterTool);
    expect(isAllowed(afterTool, EOS_TOKEN_ID)).toBe(true);

    const second = new StructuredOutputProcessor(gemmaTokenizer, responseFormat);
    const secondIds = await consume(second, "<|channel>thought\nThe tool returned 9.<channel|>The answer is ");
    const content = logits();
    second([secondIds], content);
    expect(isAllowed(content, "9".charCodeAt(0))).toBe(true);
    expect(isAllowed(content, "x".charCodeAt(0))).toBe(false);
    const finalIds = await consume(second, "9.<turn|>");
    const final = logits();
    second([finalIds], final);
    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it.each(["(?P<name>a)(?P=name)", "(?<name>a)\\k<name>", "(a)\\1", "(?<=a)b", "(?P<>a)"])("rejects unsupported capture-dependent or malformed patterns: %s", (regex) => {
    expect(() => new StructuredOutputProcessor(tokenizer, { type: "regex", regex })).toThrow(/unsupported|backreferences/);
  });

  it("applies constraints only to content when a template has tools but no thinking", async () => {
    const toolsOnlyTokenizer = {
      ...toolTokenizer,
      response_template: {
        ...toolTokenizer.response_template,
        fields: {
          content: toolTokenizer.response_template.fields.content,
          tool_calls: toolTokenizer.response_template.fields.tool_calls,
        },
      },
    };
    const toolCall = new StructuredOutputProcessor(toolsOnlyTokenizer, { type: "regex", regex: "ok" });
    const inputIds = await consume(toolCall, '<tool_call>{"unconstrained":true}</tool_call>');
    const final = logits();

    toolCall([inputIds], final);

    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it("allows an EOS token that closes a tool-call-only response", async () => {
    const encoder = new TextEncoder();
    const eosTokenId = 256;
    const eosClosingTokenizer = {
      tokens: [...Array.from({ length: 256 }, (_, tokenId) => [tokenId]), encoder.encode("</tool_call>")],
      eos_token_id: eosTokenId,
      special_token_ids: [eosTokenId],
      response_template: {
        start_anchor: "<assistant>",
        fields: {
          content: { close: "</content>", content: "text" },
          tool_calls: { open: "<tool_call>", close: "</tool_call>", content: "json" },
        },
      },
    };
    const processor = new StructuredOutputProcessor(eosClosingTokenizer, { type: "regex", regex: "ok" });
    const inputIds = await consume(processor, '<tool_call>{"unconstrained":true}');
    const scores = new Tensor("float32", new Float32Array(eosTokenId + 1).fill(1), [1, eosTokenId + 1]);

    processor([inputIds], scores);

    expect(isAllowed(scores, eosTokenId)).toBe(true);
  });

  it("does not allow EOS inside a repeated tool-call opener", async () => {
    const processor = new StructuredOutputProcessor(toolTokenizer, { type: "regex", regex: "ok" });
    const inputIds = await consume(processor, '<tool_call>{"first":true}</tool_call><');
    const scores = logits();

    processor([inputIds], scores);

    expect(isAllowed(scores, EOS_TOKEN_ID)).toBe(false);
  });

  it.each(["direct", "nested", "extended"])("forwards disabled thinking context through a %s processor list", async (kind) => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "json_object" });
    const mockTokenizer = Object.assign(() => ({ input_ids: new Tensor("int64", BigInt64Array.of(0n), [1, 1]) }), {
      apply_chat_template: () => "<assistant>",
      batch_decode: () => ["<assistant>"],
    });
    const pipe = new TextGenerationPipeline({
      task: "text-generation",
      tokenizer: mockTokenizer,
      model: { generate: async ({ input_ids }) => input_ids },
    });

    const nested = new LogitsProcessorList();
    nested.push(processor);
    const outer = new LogitsProcessorList();
    if (kind === "nested") outer.push(nested);
    else outer.extend(nested);

    await pipe([{ role: "user", content: "hi" }], {
      tokenizer_encode_kwargs: { enable_thinking: false },
      logits_processor: kind === "direct" ? processor : outer,
    });

    const scores = logits();
    processor([[0n]], scores);
    expect(isAllowed(scores, "<".charCodeAt(0))).toBe(false);
    expect(isAllowed(scores, "{".charCodeAt(0))).toBe(true);
  });

  it("waits for an explicit content opener after thinking", async () => {
    const explicitContentTokenizer = {
      ...thinkingTokenizer,
      response_template: {
        ...thinkingTokenizer.response_template,
        fields: {
          ...thinkingTokenizer.response_template.fields,
          content: { open: "<final>", close: "</final>", content: "text" },
        },
      },
    };
    const processor = new StructuredOutputProcessor(explicitContentTokenizer, {
      type: "regex",
      regex: "answer: \\d+",
    });
    const inputIds = await consume(processor, "<think>reasoning</think><final>answer: 42");
    const final = logits();

    processor([inputIds], final);

    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it("switches to the constraint within a special token", () => {
    const encoder = new TextEncoder();
    const openerTokenId = 256;
    const transitionTokenId = 257;
    const eosTokenId = 258;
    const compoundTokenizer = {
      tokens: [...Array.from({ length: 256 }, (_, tokenId) => [tokenId]), encoder.encode("<think>"), encoder.encode("</think>{"), []],
      eos_token_id: eosTokenId,
      special_token_ids: [openerTokenId, transitionTokenId, eosTokenId],
      response_template: thinkingTokenizer.response_template,
    };
    const processor = new StructuredOutputProcessor(compoundTokenizer, { type: "json_object" });
    const inputIds = [0n];
    const makeLogits = () => new Tensor("float32", new Float32Array(eosTokenId + 1).fill(1), [1, eosTokenId + 1]);

    const initial = makeLogits();
    processor([inputIds], initial);
    expect(isAllowed(initial, openerTokenId)).toBe(true);
    inputIds.push(BigInt(openerTokenId));

    const thinking = makeLogits();
    processor([inputIds], thinking);
    expect(isAllowed(thinking, transitionTokenId)).toBe(true);
    inputIds.push(BigInt(transitionTokenId));

    const structured = makeLogits();
    processor([inputIds], structured);
    expect(isAllowed(structured, '"'.charCodeAt(0))).toBe(true);
    expect(isAllowed(structured, "x".charCodeAt(0))).toBe(false);
  });

  it("continues thinking when its opener was prefilled in the prompt", async () => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "answer: \\d+" });
    const inputIds = Array.from(new TextEncoder().encode("<assistant><think>\n"), BigInt);
    const thinking = logits();

    processor([inputIds], thinking);

    expect(isAllowed(thinking, "x".charCodeAt(0))).toBe(true);
    expect(isAllowed(thinking, EOS_TOKEN_ID)).toBe(false);

    for (const tokenId of new TextEncoder().encode("reasoning</think>answer: 42")) {
      const scores = logits();
      processor([inputIds], scores);
      expect(isAllowed(scores, tokenId)).toBe(true);
      inputIds.push(BigInt(tokenId));
    }
    const final = logits();
    processor([inputIds], final);
    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it("recognizes the longest thinking delimiter across token boundaries", async () => {
    const overlappingTokenizer = {
      ...thinkingTokenizer,
      response_template: {
        ...thinkingTokenizer.response_template,
        fields: {
          ...thinkingTokenizer.response_template.fields,
          thinking: { open: "<think>", close: ["</think>", "</think>\n"], content: "text" },
        },
      },
    };
    const processor = new StructuredOutputProcessor(overlappingTokenizer, { type: "regex", regex: "ok" });
    const inputIds = await consume(processor, "<think>reasoning</think>ok");
    const final = logits();

    processor([inputIds], final);
    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);

    const extended = new StructuredOutputProcessor(overlappingTokenizer, { type: "regex", regex: "ok" });
    const extendedIds = await consume(extended, "<think>reasoning</think>\nok");
    const extendedFinal = logits();
    extended([extendedIds], extendedFinal);
    expect(isAllowed(extendedFinal, EOS_TOKEN_ID)).toBe(true);
  });

  it("blocks unrelated special tokens during thinking, including alternate end tokens", () => {
    const encoder = new TextEncoder();
    const specials = ["<think>", "</thi", "nk>", "<turn|>", "<|tool_response>", "<unrelated>", "</think>{", "</think>x", ""];
    const source = {
      tokens: [...Array.from({ length: 256 }, (_, id) => [id]), ...specials.map((text) => encoder.encode(text)), []],
      eos_token_id: 256 + specials.length,
      special_token_ids: Array.from({ length: specials.length + 1 }, (_, i) => 256 + i),
      response_template: thinkingTokenizer.response_template,
    };
    const processor = new StructuredOutputProcessor(source, { type: "json_object" });
    const input = [0n];
    const scores = () => new Tensor("float32", new Float32Array(source.tokens.length).fill(1), [1, source.tokens.length]);
    processor([input], scores());
    input.push(256n);
    const thinking = scores();
    processor([input], thinking);
    expect(isAllowed(thinking, 257)).toBe(true);
    for (const id of [259, 260, 261, 263, 264, source.eos_token_id]) expect(isAllowed(thinking, id)).toBe(false);
    expect(isAllowed(thinking, 262)).toBe(true);
    input.push(257n);
    const partial = scores();
    processor([input], partial);
    expect(isAllowed(partial, 258)).toBe(true);
    expect(isAllowed(partial, 259)).toBe(false);
    input.push(258n);
    const content = scores();
    processor([input], content);
    expect(isAllowed(content, "{".charCodeAt(0))).toBe(true);
    expect(isAllowed(content, "x".charCodeAt(0))).toBe(false);
  });

  it("rejects incompatible answer prefills instead of resetting the constraint", () => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "ok" });
    const input = Array.from(new TextEncoder().encode("<assistant>wrong"), BigInt);
    expect(() => processor([input], logits())).toThrow("prompt prefill");
  });

  it("accepts a special closer after plain reasoning ends with a false closer prefix", () => {
    const encoder = new TextEncoder();
    const closerId = 257;
    const source = {
      ...thinkingTokenizer,
      tokens: [...tokenizer.tokens, encoder.encode("</think>")],
      special_token_ids: [EOS_TOKEN_ID, closerId],
    };
    const processor = new StructuredOutputProcessor(source, { type: "regex", regex: "ok" });
    const input = Array.from(encoder.encode("<assistant><think>reasoning</thi"), BigInt);
    const makeScores = () => new Tensor("float32", new Float32Array(source.tokens.length).fill(1), [1, source.tokens.length]);
    const thinking = makeScores();
    processor([input], thinking);
    expect(isAllowed(thinking, closerId)).toBe(true);
    input.push(BigInt(closerId));
    const content = makeScores();
    processor([input], content);
    expect(isAllowed(content, "o".charCodeAt(0))).toBe(true);
    expect(isAllowed(content, "x".charCodeAt(0))).toBe(false);
  });

  it("rejects pattern start anchors instead of ignoring prompt replay", () => {
    const source = {
      ...thinkingTokenizer,
      response_template: {
        start_anchor_pattern: "<assistant>\\s*",
        fields: thinkingTokenizer.response_template.fields,
      },
    };
    expect(() => new StructuredOutputProcessor(source, { type: "json_object" })).toThrow("start_anchor_pattern");
  });

  it("replays completed disabled-thinking prefills before applying constraints", () => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "ok" });
    processor.setGenerationContext({ enable_thinking: false });
    const input = Array.from(new TextEncoder().encode("<assistant><think>\n</think>"), BigInt);
    const scores = logits();
    processor([input], scores);
    expect(isAllowed(scores, "o".charCodeAt(0))).toBe(true);
    expect(isAllowed(scores, "<".charCodeAt(0))).toBe(false);
    expect(() => processor.setGenerationContext({ enable_thinking: true })).toThrow("before generation");
  });

  it.each([false, true])("ignores prompt framing whitespace after a completed thinking block (thinking=%s)", (enable_thinking) => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "ok" });
    processor.setGenerationContext({ enable_thinking });
    const input = Array.from(new TextEncoder().encode("<assistant><think>\n\n</think>\n\n"), BigInt);
    const scores = logits();
    processor([input], scores);
    expect(isAllowed(scores, "o".charCodeAt(0))).toBe(true);
    expect(isAllowed(scores, "\n".charCodeAt(0))).toBe(false);
  });

  it("still validates payload prefills after thinking-block framing whitespace", () => {
    const encoder = new TextEncoder();
    const valid = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "ok" });
    const scores = logits();
    valid([Array.from(encoder.encode("<assistant><think></think>\n\no"), BigInt)], scores);
    expect(isAllowed(scores, "k".charCodeAt(0))).toBe(true);

    const invalid = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "ok" });
    expect(() => invalid([Array.from(encoder.encode("<assistant><think></think>\n\nx"), BigInt)], logits())).toThrow("prompt prefill");
  });

  it("rejects unfinished thinking prefills when thinking is explicitly disabled", () => {
    const processor = new StructuredOutputProcessor(thinkingTokenizer, { type: "regex", regex: "ok" });
    processor.setGenerationContext({ enable_thinking: false });
    const input = Array.from(new TextEncoder().encode("<assistant><think>"), BigInt);
    expect(() => processor([input], logits())).toThrow("unfinished thinking");
  });

  it("does not share close-plus-content masks across different grammars", () => {
    const encoder = new TextEncoder();
    const source = {
      ...thinkingTokenizer,
      tokens: [...tokenizer.tokens, encoder.encode("</think>a"), encoder.encode("</think>b")],
    };
    const input = Array.from(encoder.encode("<assistant><think>"), BigInt);
    const makeScores = () => new Tensor("float32", new Float32Array(source.tokens.length).fill(1), [1, source.tokens.length]);
    for (const [regex, allowed, blocked] of [
      ["a", 257, 258],
      ["b", 258, 257],
    ]) {
      const processor = new StructuredOutputProcessor(source, { type: "regex", regex });
      const scores = makeScores();
      processor([input], scores);
      expect(isAllowed(scores, allowed)).toBe(true);
      expect(isAllowed(scores, blocked)).toBe(false);
    }
  });

  it("accepts a supported response-template thinking opener pattern", async () => {
    const patternedTokenizer = {
      ...thinkingTokenizer,
      response_template: {
        ...thinkingTokenizer.response_template,
        fields: {
          ...thinkingTokenizer.response_template.fields,
          thinking: { open_pattern: "<think>\\s*", close: "</think>", content: "text" },
        },
      },
    };
    const processor = new StructuredOutputProcessor(patternedTokenizer, { type: "regex", regex: "ok" });
    const inputIds = await consume(processor, "<think>\nreasoning</think>ok");
    const final = logits();

    processor([inputIds], final);
    expect(isAllowed(final, EOS_TOKEN_ID)).toBe(true);
  });

  it("rejects unsupported thinking close patterns instead of constraining reasoning", () => {
    const unsupportedTokenizer = {
      ...thinkingTokenizer,
      response_template: {
        ...thinkingTokenizer.response_template,
        fields: {
          ...thinkingTokenizer.response_template.fields,
          thinking: { open: "<think>", close_pattern: "</think>\\s*", content: "text" },
        },
      },
    };

    expect(() => new StructuredOutputProcessor(unsupportedTokenizer, { type: "json_object" })).toThrow("thinking close_pattern");
  });

  it("allows a response-template content closer only after a complete answer", async () => {
    const encoder = new TextEncoder();
    const turnTokenId = 256;
    const eosTokenId = 257;
    const endingTokenizer = {
      tokens: [...Array.from({ length: 256 }, (_, tokenId) => [tokenId]), encoder.encode("<turn|>"), []],
      eos_token_id: eosTokenId,
      special_token_ids: [turnTokenId, eosTokenId],
      response_template: {
        ...thinkingTokenizer.response_template,
        fields: {
          ...thinkingTokenizer.response_template.fields,
          content: { close: ["<turn|>", "<eos>"], content: "text" },
        },
      },
    };
    const makeLogits = () => new Tensor("float32", new Float32Array(eosTokenId + 1).fill(1), [1, eosTokenId + 1]);
    const processor = new StructuredOutputProcessor(endingTokenizer, { type: "json_object" });
    const inputIds = [0n];
    processor([inputIds], makeLogits());

    for (const tokenId of encoder.encode('<think>reasoning</think>{"answer":391')) {
      inputIds.push(BigInt(tokenId));
    }
    const incomplete = makeLogits();
    processor([inputIds], incomplete);
    expect(isAllowed(incomplete, turnTokenId)).toBe(false);

    inputIds.push(BigInt("}".charCodeAt(0)));
    const complete = makeLogits();
    processor([inputIds], complete);
    expect(isAllowed(complete, turnTokenId)).toBe(true);
    expect(isAllowed(complete, eosTokenId)).toBe(true);
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
