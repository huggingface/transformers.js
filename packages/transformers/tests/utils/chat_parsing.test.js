import { parseResponse, ResponseParser } from "../../src/utils/chat_parsing/index.js";

const gemma4Template = {
  defaults: { role: "assistant" },
  start_anchor: ["<|turn>model\n", "<tool_response|>"],
  fields: {
    thinking: {
      open: "<|channel>thought\n",
      close: "<channel|>",
      content: "text",
    },
    tool_calls: {
      open_pattern: String.raw`<\|tool_call>call:(?P<name>\w+)`,
      close: "<tool_call|>",
      repeats: true,
      content: "json",
      content_args: {
        unquoted_keys: true,
        string_delims: [['<|"|>', '<|"|>']],
      },
      transform: {
        type: "function",
        function: { name: "{name}", arguments: "{content}" },
      },
    },
    content: {
      close: ["<turn|>", "<|tool_response>", "<eos>"],
      content: "text",
    },
  },
};

const parseGemma4 = (text, options = {}) => parseResponse(text, gemma4Template, { prefix: "<|turn>model\n", ...options });

// onnx-community/granite-4.0-micro-ONNX
const granite4Template = {
  defaults: { role: "assistant" },
  start_anchor: "<|start_of_role|>assistant<|end_of_role|>",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open: "<tool_call>",
      close: "</tool_call>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: "{content}" },
    },
    content: { close: "<|end_of_text|>", content: "text" },
  },
};

// onnx-community/LFM2.5-350M-ONNX
const lfm25Template = {
  defaults: { role: "assistant" },
  start_anchor: "<|im_start|>assistant\n",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    content: { close: "<|im_end|>", content: "text" },
  },
};

describe("Response templates", () => {
  it("parses onnx-community/granite-4.0-micro-ONNX responses", () => {
    const prefix = "<|start_of_role|>user<|end_of_role|>What is the weather?<|end_of_text|>\n" + "<|start_of_role|>assistant<|end_of_role|>";
    const response = '<think>Use the weather tool.</think><tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call><|end_of_text|>';

    expect(parseResponse(response, granite4Template, { prefix })).toEqual({
      role: "assistant",
      thinking: "Use the weather tool.",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { city: "Paris" } },
        },
      ],
    });
  });

  it("parses onnx-community/LFM2.5-350M-ONNX responses", () => {
    const prefix = "<|startoftext|><|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
    const response = "<think>I should answer directly.</think>Paris.<|im_end|>";

    expect(parseResponse(response, lfm25Template, { prefix })).toEqual({
      role: "assistant",
      thinking: "I should answer directly.",
      content: "Paris.",
    });
  });

  it("parses a complete thinking response", () => {
    expect(parseGemma4("<|channel>thought\nI should answer briefly.<channel|>Paris<turn|>")).toEqual({
      role: "assistant",
      thinking: "I should answer briefly.",
      content: "Paris",
    });
  });

  it("parses an unfinished thinking region", () => {
    expect(parseGemma4("<|channel>thought\nI should check", { partial: true })).toEqual({
      role: "assistant",
      thinking: "I should check",
    });
  });

  it("holds an unfinished delimiter out of the parsed text", () => {
    expect(parseGemma4("<|channel>thought\nI should check<chan", { partial: true })).toEqual({
      role: "assistant",
      thinking: "I should check",
    });
  });

  it("holds the literal prefix of an unfinished regex delimiter", () => {
    expect(parseGemma4("<|channel>thought\nUse a tool.<channel|><|tool_ca", { partial: true })).toEqual({
      role: "assistant",
      thinking: "Use a tool.",
    });
  });

  it("recognizes a thinking close marker whose opener was prefilled", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<assistant>",
      fields: {
        thinking: { open: "<think>", close: "</think>", content: "text" },
        content: { content: "text" },
      },
    };
    expect(parseResponse("answer", template, { prefix: "history<assistant><think>prefilled reasoning</think>" })).toEqual({
      role: "assistant",
      thinking: "prefilled reasoning",
      content: "answer",
    });
  });

  it("parses accumulated streamer output at every step", () => {
    let response = "";
    const parsed = [];
    for (const chunk of ["<|channel>thought\nI should", " answer.<channel|>", "Paris"]) {
      response += chunk;
      parsed.push(parseGemma4(response, { partial: true }));
    }
    expect(parsed).toEqual([
      { role: "assistant", thinking: "I should" },
      { role: "assistant", thinking: "I should answer." },
      { role: "assistant", thinking: "I should answer.", content: "Paris" },
    ]);
  });

  it("parses incremental chunks without reparsing accumulated text", () => {
    const parser = new ResponseParser(gemma4Template, { prefix: "<|turn>model\n" });
    expect(parser.feed("<|channel>thought\nI should")).toEqual({
      role: "assistant",
      thinking: "I should",
    });
    expect(parser.feed(" answer.<channel")).toEqual({
      role: "assistant",
      thinking: "I should answer.",
    });
    expect(parser.feed("|>Paris")).toEqual({
      role: "assistant",
      thinking: "I should answer.",
      content: "Paris",
    });
    expect(parser.finalize()).toEqual({
      role: "assistant",
      thinking: "I should answer.",
      content: "Paris",
    });
    expect(() => parser.feed("more")).toThrow("already finalized");
    expect(() => parser.finalize()).toThrow("already finalized");
  });

  it("enforces required fields only when an incremental parser is finalized", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: { answer: { open: "<answer>", close: "</answer>", optional: false } },
    };
    const parser = new ResponseParser(template, { prefix: "" });
    expect(parser.feed("")).toEqual({});
    expect(() => parser.finalize()).toThrow("Required response_template fields missing from parsed output: answer");
  });

  it("does not commit growing regex captures at chunk boundaries", () => {
    const parser = new ResponseParser(gemma4Template, { prefix: "<|turn>model\n" });
    expect(parser.feed("<|tool_call>call:get_wea")).toEqual({ role: "assistant" });
    expect(parser.feed('ther{city:<|"|>Paris<|"|>}<tool_call|>')).toEqual({
      role: "assistant",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { city: "Paris" } },
        },
      ],
    });
    expect(parser.finalize()).toEqual({
      role: "assistant",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { city: "Paris" } },
        },
      ],
    });
  });

  it("returns immutable snapshots with structural sharing", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: {
        items: { open: "<item>", close: "</item>", content: "json", repeats: true },
      },
    };
    const parser = new ResponseParser(template, { prefix: "" });
    const first = parser.feed('<item>{"value":1}</item>');
    const second = parser.feed('<item>{"value":2}</item>');

    expect(first).toEqual({ items: [{ value: 1 }] });
    expect(second).toEqual({ items: [{ value: 1 }, { value: 2 }] });
    expect(second.items).not.toBe(first.items);
    expect(second.items[0]).toBe(first.items[0]);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.items)).toBe(true);
    expect(Object.isFrozen(first.items[0])).toBe(true);
    expect(() => (first.items[0].value = 3)).toThrow();
    expect(parser.finalize()).toEqual(second);
  });

  it("omits incomplete structured fields until they close", () => {
    const partial = '<|channel>thought\nUse weather.<channel|><|tool_call>call:get_weather{city:<|"|>Par';
    expect(parseGemma4(partial, { partial: true })).toEqual({
      role: "assistant",
      thinking: "Use weather.",
    });

    expect(parseGemma4(`${partial}is<|"|>}<tool_call|>`)).toEqual({
      role: "assistant",
      thinking: "Use weather.",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { city: "Paris" } },
        },
      ],
    });
  });

  it("returns plain content when no explicit region is generated", () => {
    expect(parseGemma4("Paris")).toEqual({ role: "assistant", content: "Paris" });
  });

  it("parses a complete implicit JSON field", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<assistant>",
      fields: { content: { content: "json" } },
    };
    expect(parseResponse('{"city":"Paris"}', template, { prefix: "" })).toEqual({
      role: "assistant",
      content: { city: "Paris" },
    });
    expect(parseResponse('{"city":', template, { prefix: "", partial: true })).toEqual({ role: "assistant" });
  });

  it("does not mutate array defaults across repeated parsing", () => {
    const template = {
      defaults: { role: "assistant", tool_calls: [] },
      start_anchor: "<assistant>",
      fields: {
        tool_calls: {
          open: "<tool>",
          close: "</tool>",
          repeats: true,
          content: "json",
        },
      },
    };
    const response = '<tool>{"name":"weather"}</tool>';
    expect(parseResponse(response, template, { prefix: "" }).tool_calls).toEqual([{ name: "weather" }]);
    expect(parseResponse(response, template, { prefix: "" }).tool_calls).toEqual([{ name: "weather" }]);
    expect(template.defaults.tool_calls).toEqual([]);
  });

  it("ignores zero-width regex anchors without hanging", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<assistant>",
      fields: {
        marker: { open_pattern: "(?=x)", close: "</marker>", content: "text" },
        content: { content: "text" },
      },
    };
    expect(parseResponse("x", template, { prefix: "" })).toEqual({ role: "assistant", content: "x" });
  });

  it("parses whitespace around lax JSON keys", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<assistant>",
      fields: {
        tool_calls: {
          open: "<tool>",
          close: "</tool>",
          content: "json",
          content_args: { unquoted_keys: true },
        },
      },
    };
    expect(parseResponse('<tool>{ city: "Paris", days : 3 }</tool>', template, { prefix: "" })).toEqual({
      role: "assistant",
      tool_calls: { city: "Paris", days: 3 },
    });
  });

  it("omits malformed partial booleans", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<assistant>",
      fields: { content: { content: "bool" } },
    };
    expect(parseResponse("f", template, { prefix: "", partial: true })).toEqual({ role: "assistant" });
    expect(parseResponse("false", template, { prefix: "" })).toEqual({ role: "assistant", content: false });
  });

  it("uses a prompt-prefilled opener for partial output", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<assistant>",
      fields: {
        thinking: { open: "<think>", close: "</think>", content: "text" },
        content: { content: "text" },
      },
    };
    expect(parseResponse("still reasoning", template, { prefix: "history<assistant><think>", partial: true })).toEqual({
      role: "assistant",
      thinking: "still reasoning",
    });
  });

  it("requires explicit prompt and start-anchor context", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: { content: { content: "text" } },
    };
    expect(() => parseResponse("answer", template)).toThrow("requires a string prefix");
    expect(() => parseResponse("answer", { fields: template.fields }, { prefix: "" })).toThrow("must define 'start_anchor' or 'start_anchor_pattern'");
  });

  it("throws for malformed completed structured fields", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: {
        tool: { open: "<tool>", close: "</tool>", content: "json" },
      },
    };
    expect(() => parseResponse("<tool>{bad}</tool>", template, { prefix: "", partial: true })).toThrow("Could not parse response field as JSON");
  });

  it("enforces required fields only for final responses", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: {
        answer: { open: "<answer>", close: "</answer>", optional: false },
      },
    };
    expect(parseResponse("", template, { prefix: "", partial: true })).toEqual({});
    expect(() => parseResponse("", template, { prefix: "" })).toThrow("Required response_template fields missing from parsed output: answer");
  });

  it("closes structured fields with an end anchor on finalization", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: {
        data: { open: "<json>", close_pattern: String.raw`\Z`, content: "json", optional: false },
      },
    };
    expect(parseResponse('<json>{"city":"Paris"}', template, { prefix: "" })).toEqual({
      data: { city: "Paris" },
    });
  });

  it("defines model-generated keys without changing object prototypes", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: {
        content: { content: "kv-lines", content_args: { kv_sep: "=" } },
      },
    };
    const parsed = parseResponse("__proto__=safe", template, { prefix: "" });
    expect(Object.getPrototypeOf(parsed.content)).toBe(Object.prototype);
    expect(Object.hasOwn(parsed.content, "__proto__")).toBe(true);
    expect(parsed.content.__proto__).toBe("safe");
  });

  it("supports field names that shadow object prototype properties", () => {
    const fields = JSON.parse('{"__proto__":{"open":"<x>","close":"</x>","optional":false}}');
    const parsed = parseResponse("<x>value</x>", { start_anchor: "<assistant>", fields }, { prefix: "" });
    expect(Object.getPrototypeOf(parsed)).toBe(Object.prototype);
    expect(Object.hasOwn(parsed, "__proto__")).toBe(true);
    expect(parsed.__proto__).toBe("value");
  });

  it("rejects integers that cannot be represented precisely", () => {
    const template = {
      start_anchor: "<assistant>",
      fields: { content: { content: "int" } },
    };
    expect(() => parseResponse("9007199254740993", template, { prefix: "" })).toThrow("without losing integer precision");
  });
});
