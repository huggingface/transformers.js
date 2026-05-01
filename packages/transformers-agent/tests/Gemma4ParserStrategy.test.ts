import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4ParserStrategy } from "../src/parsers/Gemma4ParserStrategy";

function parse(content: string) {
  let id = 0;
  return new Gemma4ParserStrategy().parseAssistantContent(content, (prefix) => `${prefix}_${++id}`);
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
