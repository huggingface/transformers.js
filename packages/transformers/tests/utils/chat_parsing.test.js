import { parseResponse } from "../../src/utils/chat_parsing/index.js";

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

describe("Response templates", () => {
  it("parses a complete thinking response", () => {
    expect(parseResponse("<|channel>thought\nI should answer briefly.<channel|>Paris<turn|>", gemma4Template)).toEqual({
      role: "assistant",
      thinking: "I should answer briefly.",
      content: "Paris",
    });
  });

  it("parses an unfinished thinking region", () => {
    expect(parseResponse("<|channel>thought\nI should check", gemma4Template)).toEqual({
      role: "assistant",
      thinking: "I should check",
    });
  });

  it("holds an unfinished delimiter out of the parsed text", () => {
    expect(parseResponse("<|channel>thought\nI should check<chan", gemma4Template)).toEqual({
      role: "assistant",
      thinking: "I should check",
    });
  });

  it("holds the literal prefix of an unfinished regex delimiter", () => {
    expect(parseResponse("<|channel>thought\nUse a tool.<channel|><|tool_ca", gemma4Template)).toEqual({
      role: "assistant",
      thinking: "Use a tool.",
    });
  });

  it("recognizes a thinking close marker whose opener was prefilled", () => {
    const template = {
      defaults: { role: "assistant" },
      fields: {
        thinking: { open: "<think>", close: "</think>", content: "text" },
        content: { content: "text" },
      },
    };
    expect(parseResponse("prefilled reasoning</think>answer", template)).toEqual({
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
      parsed.push(parseResponse(response, gemma4Template));
    }
    expect(parsed).toEqual([
      { role: "assistant", thinking: "I should" },
      { role: "assistant", thinking: "I should answer." },
      { role: "assistant", thinking: "I should answer.", content: "Paris" },
    ]);
  });

  it("omits incomplete structured fields until they close", () => {
    const partial = '<|channel>thought\nUse weather.<channel|><|tool_call>call:get_weather{city:<|"|>Par';
    expect(parseResponse(partial, gemma4Template)).toEqual({
      role: "assistant",
      thinking: "Use weather.",
    });

    expect(parseResponse(`${partial}is<|"|>}<tool_call|>`, gemma4Template)).toEqual({
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
    expect(parseResponse("Paris", gemma4Template)).toEqual({ role: "assistant", content: "Paris" });
  });

  it("parses a complete implicit JSON field", () => {
    const template = {
      defaults: { role: "assistant" },
      fields: { content: { content: "json" } },
    };
    expect(parseResponse('{"city":"Paris"}', template)).toEqual({
      role: "assistant",
      content: { city: "Paris" },
    });
    expect(parseResponse('{"city":', template)).toEqual({ role: "assistant" });
  });

  it("does not mutate array defaults across repeated parsing", () => {
    const template = {
      defaults: { role: "assistant", tool_calls: [] },
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
    expect(parseResponse(response, template).tool_calls).toEqual([{ name: "weather" }]);
    expect(parseResponse(response, template).tool_calls).toEqual([{ name: "weather" }]);
    expect(template.defaults.tool_calls).toEqual([]);
  });

  it("ignores zero-width regex anchors without hanging", () => {
    const template = {
      defaults: { role: "assistant" },
      fields: {
        marker: { open_pattern: "(?=x)", close: "</marker>", content: "text" },
        content: { content: "text" },
      },
    };
    expect(parseResponse("x", template)).toEqual({ role: "assistant", content: "x" });
  });

  it("parses whitespace around lax JSON keys", () => {
    const template = {
      defaults: { role: "assistant" },
      fields: {
        tool_calls: {
          open: "<tool>",
          close: "</tool>",
          content: "json",
          content_args: { unquoted_keys: true },
        },
      },
    };
    expect(parseResponse('<tool>{ city: "Paris", days : 3 }</tool>', template)).toEqual({
      role: "assistant",
      tool_calls: { city: "Paris", days: 3 },
    });
  });

  it("omits malformed partial booleans", () => {
    const template = {
      defaults: { role: "assistant" },
      fields: { content: { content: "bool" } },
    };
    expect(parseResponse("f", template)).toEqual({ role: "assistant" });
    expect(parseResponse("false", template)).toEqual({ role: "assistant", content: false });
  });
});
