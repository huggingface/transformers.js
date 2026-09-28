import { parseResponse, ResponseParser } from "../../src/utils/chat_parsing/index.js";

const smollmTemplate = {
  defaults: { role: "assistant" },
  start_anchor: "<|im_start|>assistant\n",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open: "<tool_call>",
      close: "</tool_call>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: "{content}" },
    },
    content: { close: "<|im_end|>", content: "text" },
  },
};

const gemma4Template = {
  defaults: { role: "assistant" },
  start_anchor: ["<|turn>model\n", "<tool_response|>"],
  fields: {
    thinking: { open: "<|channel>thought\n", close: "<channel|>", content: "text" },
    tool_calls: {
      open_pattern: String.raw`<\|tool_call>call:(?P<name>\w+)`,
      close: "<tool_call|>",
      repeats: true,
      content: "json",
      content_args: {
        unquoted_keys: true,
        string_delims: [['<|"|>', '<|"|>']],
      },
      transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
    },
    content: { close: ["<turn|>", "<|tool_response>", "<eos>"], content: "text" },
  },
};

describe("Response templates", () => {
  it("parses thinking and content", () => {
    const output = parseResponse("<think>Consider gravity.</think>Gravity attracts masses.<|im_end|>", smollmTemplate, { prefix: "" });
    expect(output).toEqual({
      role: "assistant",
      thinking: "Consider gravity.",
      content: "Gravity attracts masses.",
    });
  });

  it("uses a prompt-prefilled thinking opener", () => {
    const output = parseResponse("Continue thinking.</think>Final answer.<|im_end|>", smollmTemplate, {
      prefix: "old history<|im_start|>assistant\n<think>",
    });
    expect(output).toEqual({
      role: "assistant",
      thinking: "Continue thinking.",
      content: "Final answer.",
    });
  });

  it("parses Gemma 4 thinking and lax-JSON tool calls", () => {
    const output = parseResponse('<|channel>thought\nI should call the tool.<channel|><|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|><|tool_response>', gemma4Template, { prefix: "" });
    expect(output).toEqual({
      role: "assistant",
      thinking: "I should call the tool.",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { city: "Paris", days: 3 } },
        },
      ],
    });
  });

  it("parses regex-delimited XML tool arguments", () => {
    const template = {
      defaults: { role: "assistant" },
      start_anchor: "<|im_start|>assistant\n",
      fields: {
        tool_calls: {
          open_pattern: String.raw`<tool_call>\s*<function=(?P<name>\w+)>`,
          close: "</tool_call>",
          repeats: true,
          content: "xml-inline",
          content_args: {
            tag_pattern: String.raw`<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>`,
            value_parser: { name: "json", args: { allow_non_json: true } },
          },
          transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
        },
      },
    };
    const output = parseResponse('<tool_call>\n<function=get_weather><parameter=cities>["Paris","Tokyo"]</parameter><parameter=unit>celsius</parameter></function></tool_call>', template, { prefix: "" });
    expect(output).toEqual({
      role: "assistant",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { cities: ["Paris", "Tokyo"], unit: "celsius" } },
        },
      ],
    });
  });

  it("coerces tool arguments using their JSON schema", () => {
    const output = parseResponse('<tool_call>{"name":"weather","arguments":{"days":"3","metric":"true"}}</tool_call>', smollmTemplate, {
      prefix: "",
      tools: [
        {
          type: "function",
          function: {
            name: "weather",
            parameters: {
              type: "object",
              properties: { days: { type: "integer" }, metric: { type: "boolean" } },
            },
          },
        },
      ],
    });
    expect(output.tool_calls[0].function.arguments).toEqual({ days: 3, metric: true });
  });

  it("emits streaming events across literal delimiter boundaries", () => {
    const parser = new ResponseParser(smollmTemplate, { prefix: "<|im_start|>assistant\n" });
    const events = [...parser.feed("<thi"), ...parser.feed("nk>work</thi"), ...parser.feed("nk>answer"), ...parser.finalize()[1]];
    expect(events).toEqual([
      { type: "region_open", field: "thinking" },
      { type: "region_chunk", field: "thinking", text: "work", dirty: false },
      { type: "region_close", field: "thinking", value: "work" },
      { type: "region_open", field: "content" },
      { type: "region_chunk", field: "content", text: "answer", dirty: false },
      { type: "region_close", field: "content", value: "answer" },
    ]);
  });

  it("validates malformed templates", () => {
    expect(() => parseResponse("x", { version: 2, start_anchor: "x", fields: { content: {} } }, { prefix: "" })).toThrow("Unsupported response_template version");
    expect(() => parseResponse("x", { start_anchor: "x", fields: { first: {}, second: {} } }, { prefix: "" })).toThrow("at most one field without an opener");
  });
});
