import assert from "node:assert/strict";
import test from "node:test";
import { ParserStrategyGranite } from "../src/parsers/ParserStrategyGranite";

function parse(content: string) {
  let id = 0;
  return new ParserStrategyGranite().parseAssistantContent(content, (prefix) => `${prefix}_${++id}`);
}

test("parses Granite-style JSON tool calls", () => {
  const result = parse('<tool_call>\n{"name": "getWeather", "arguments": {"location": "Bern"}}\n</tool_call>');

  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("does not expose incomplete Granite tool calls as visible text", () => {
  const chunks = ["<", "<tool", "<tool_call", "<tool_call>", '<tool_call>\n{"name": ', '<tool_call>\n{"name": "getWeather", ', '<tool_call>\n{"name": "getWeather", "arguments": ', '<tool_call>\n{"name": "getWeather", "arguments": {"location": ', '<tool_call>\n{"name": "getWeather", "arguments": {"location": "Bern"}}\n', '<tool_call>\n{"name": "getWeather", "arguments": {"location": "Bern"}}\n<', '<tool_call>\n{"name": "getWeather", "arguments": {"location": "Bern"}}\n</', '<tool_call>\n{"name": "getWeather", "arguments": {"location": "Bern"}}\n</tool_call'];

  for (const chunk of chunks) {
    const result = parse(chunk);

    assert.equal(result.visibleText, "");
    assert.deepEqual(result.toolCalls, []);
  }
});

test("strips Granite end-of-text token after tool calls", () => {
  const result = parse('<tool_call>\n{"name": "getWeather", "arguments": {"location": "Bern"}}\n</tool_call><|end_of_text|>');

  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("strips Granite end-of-text token from visible answers", () => {
  const result = parse("The weather in Bern is currently sunny with a temperature of 21°C.<|end_of_text|>");

  assert.equal(result.visibleText, "The weather in Bern is currently sunny with a temperature of 21°C.");
  assert.deepEqual(result.toolCalls, []);
});
