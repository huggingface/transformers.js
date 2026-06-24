import assert from "node:assert/strict";
import test from "node:test";
import { ModelAdapterGemma4 } from "../src/adapters/ModelAdapterGemma4";
import type { Message } from "../src/types";

function parse(content: string) {
  let id = 0;
  return new ModelAdapterGemma4().parseAssistantContent(content, (prefix) => `${prefix}_${++id}`);
}

test("parses a completed Gemma4 tool call with quoted string arguments", () => {
  const result = parse('<|tool_call>call:getWeather{location:<|"|>Bern<|"|>}<tool_call|>');

  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("does not parse incomplete streaming tool calls", () => {
  const chunks = ["<|tool_call>", "<|tool_call>call:getWeather{location:", '<|tool_call>call:getWeather{location:<|"|>', '<|tool_call>call:getWeather{location:<|"|>Bern', '<|tool_call>call:getWeather{location:<|"|>Bern<|"|>', '<|tool_call>call:getWeather{location:<|"|>Bern<|"|>}'];

  for (const chunk of chunks) {
    const result = parse(chunk);

    assert.equal(result.visibleText, "");
    assert.deepEqual(result.toolCalls, []);
  }
});

test("parses a completed tool call with trailing tool response marker", () => {
  const result = parse('<|tool_call>call:getWeather{location:<|"|>Bern<|"|>}<tool_call|><|tool_response>');

  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("keeps partial visible answer text while streaming", () => {
  const chunks = [
    ["The ", "The"],
    ["The weather ", "The weather"],
    ["The weather in ", "The weather in"],
    ["The weather in Bern ", "The weather in Bern"],
    ["The weather in Bern is ", "The weather in Bern is"],
    ["The weather in Bern is Sunny ", "The weather in Bern is Sunny"],
    ["The weather in Bern is Sunny with ", "The weather in Bern is Sunny with"],
    ["The weather in Bern is Sunny with a ", "The weather in Bern is Sunny with a"],
    ["The weather in Bern is Sunny with a temperature ", "The weather in Bern is Sunny with a temperature"],
    ["The weather in Bern is Sunny with a temperature of ", "The weather in Bern is Sunny with a temperature of"],
    ["The weather in Bern is Sunny with a temperature of 21°C.", "The weather in Bern is Sunny with a temperature of 21°C."],
    ["The weather in Bern is Sunny with a temperature of 21°C.<turn|>", "The weather in Bern is Sunny with a temperature of 21°C."],
  ];

  for (const [chunk, expected] of chunks) {
    const result = parse(chunk);

    assert.equal(result.visibleText, expected);
    assert.deepEqual(result.toolCalls, []);
  }
});

test("surfaces partial thinking text while streaming", () => {
  const chunks = [
    ["<|channel>thought\n1.  **Analyze the user's request:** The user is asking \"Whats ", "1.  **Analyze the user's request:** The user is asking \"Whats"],
    ["<|channel>thought\n1.  **Analyze the user's request:** The user is asking \"Whats the ", "1.  **Analyze the user's request:** The user is asking \"Whats the"],
    ["<|channel>thought\n1.  **Analyze the user's request:** The user is asking \"Whats the weather ", "1.  **Analyze the user's request:** The user is asking \"Whats the weather"],
    ["<|channel>thought\n1.  **Analyze the user's request:** The user is asking \"Whats the weather in ", "1.  **Analyze the user's request:** The user is asking \"Whats the weather in"],
    ['<|channel>thought\n1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n', '1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".'],
    ['<|channel>thought\n1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n2.  ', '1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n2.'],
    ['<|channel>thought\n1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n2.  **Identify ', '1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n2.  **Identify'],
    ['<|channel>thought\n1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n2.  **Identify the ', '1.  **Analyze the user\'s request:** The user is asking "Whats the weather in Bern?".\n2.  **Identify the'],
  ];

  for (const [chunk, expected] of chunks) {
    const result = parse(chunk);

    assert.equal(result.thinkingText, expected);
    assert.equal(result.visibleText, "");
    assert.deepEqual(result.toolCalls, []);
  }
});

test("strips Gemma4 control tokens from visible text", () => {
  const result = parse("The weather in Bern is Sunny with a temperature of 21°C.<turn|>");

  assert.equal(result.visibleText, "The weather in Bern is Sunny with a temperature of 21°C.");
  assert.deepEqual(result.toolCalls, []);
});

test("separates thought blocks from visible text and tool calls", () => {
  const result = parse('<|channel>thought\nUse getWeather for Bern.<channel|><|tool_call>call:getWeather{location:<|"|>Bern<|"|>}<tool_call|>');

  assert.equal(result.thinkingText, "Use getWeather for Bern.");
  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("formats structured tool responses for Gemma4", () => {
  const adapter = new ModelAdapterGemma4();
  const messages: Message[] = [
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "toolcall_1",
          type: "function",
          function: {
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
    },
    {
      role: "tool",
      toolCallId: "toolcall_1",
      name: "get_weather",
      content: JSON.stringify({ location: "London", temperature: 20, weather: "sunny" }),
    },
  ];

  assert.deepEqual(adapter.formatMessages(messages), [
    {
      role: "assistant",
      tool_calls: [
        {
          function: {
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
      tool_responses: [
        {
          name: "get_weather",
          response: { location: "London", temperature: 20, weather: "sunny" },
        },
      ],
    },
  ]);
});

test("normalizes text weather tool results for Gemma4", () => {
  const adapter = new ModelAdapterGemma4();
  const messages: Message[] = [
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "toolcall_1",
          type: "function",
          function: {
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
    },
    {
      role: "tool",
      toolCallId: "toolcall_1",
      name: "get_weather",
      content: "The weather in London in Sunny, 20 degrees celsius.",
    },
  ];

  assert.deepEqual(adapter.formatMessages(messages), [
    {
      role: "assistant",
      tool_calls: [
        {
          function: {
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
      tool_responses: [
        {
          name: "get_weather",
          response: { location: "London", temperature: 20, weather: "sunny" },
        },
      ],
    },
  ]);
});

test("reconstructs Gemma4 thinking content", () => {
  const adapter = new ModelAdapterGemma4();
  const messages: Message[] = [
    {
      role: "assistant",
      content: "",
      thinking: "Use get_weather for London.",
      toolCalls: [
        {
          id: "toolcall_1",
          type: "function",
          function: {
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
    },
    {
      role: "tool",
      toolCallId: "toolcall_1",
      name: "get_weather",
      content: JSON.stringify({ location: "London", temperature: 20, weather: "sunny" }),
    },
  ];

  assert.deepEqual(adapter.formatMessages(messages), [
    {
      role: "assistant",
      content: '<|channel>thought\nUse get_weather for London.<channel|><|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|><|tool_response>response:get_weather{location:<|"|>London<|"|>,temperature:20,weather:<|"|>sunny<|"|>}<tool_response|>',
    },
  ]);
});
