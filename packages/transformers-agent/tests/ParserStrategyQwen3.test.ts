import assert from "node:assert/strict";
import test from "node:test";
import { ParserStrategyQwen3 } from "../src/parsers/ParserStrategyQwen3";

function parse(content: string) {
  let id = 0;
  return new ParserStrategyQwen3().parseAssistantContent(content, (prefix) => `${prefix}_${++id}`);
}

function parseWithThinking(content: string) {
  let id = 0;
  const parser = new ParserStrategyQwen3();
  parser.supports({ modelId: "Qwen/Qwen3-8B", modelType: "qwen3_text", enableThinking: true });
  return parser.parseAssistantContent(content, (prefix) => `${prefix}_${++id}`);
}

test("parses Qwen3 tool calls with thinking text", () => {
  const result = parse(`The user is asking about the weather in Bern. Bern is a city in Switzerland, so I should use the getWeather function with "Bern" as the location parameter.
</think>

<tool_call>
<function=getWeather>
<parameter=location>
Bern
</parameter>
</function>
</tool_call><|im_end|>`);

  assert.equal(result.thinkingText, 'The user is asking about the weather in Bern. Bern is a city in Switzerland, so I should use the getWeather function with "Bern" as the location parameter.');
  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("parses Qwen3 tool calls without thinking text", () => {
  const result = parse(`<tool_call>
<function=getWeather>
<parameter=location>
Bern
</parameter>
</function>
</tool_call><|im_end|>`);

  assert.equal(result.thinkingText, "");
  assert.equal(result.visibleText, "");
  assert.deepEqual(result.toolCalls, [
    {
      id: "toolcall_1",
      name: "getWeather",
      args: { location: "Bern" },
    },
  ]);
});

test("strips Qwen3 end token from plain answers", () => {
  const result = parse("I am Qwen3.5, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, including answering questions, creating text, performing logical reasoning, coding, and more. How can I help you?<|im_end|>");

  assert.equal(result.thinkingText, "");
  assert.equal(result.visibleText, "I am Qwen3.5, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, including answering questions, creating text, performing logical reasoning, coding, and more. How can I help you?");
  assert.deepEqual(result.toolCalls, []);
});

test("surfaces partial Qwen3 thinking before closing think token", () => {
  const chunks = [
    ["The user is asking about ", "The user is asking about"],
    ["The user is asking about the weather in Bern. ", "The user is asking about the weather in Bern."],
    ['The user is asking about the weather in Bern. Bern is a city in Switzerland, so I should use the getWeather function with "Bern" as the location parameter.', 'The user is asking about the weather in Bern. Bern is a city in Switzerland, so I should use the getWeather function with "Bern" as the location parameter.'],
  ];

  for (const [chunk, expected] of chunks) {
    const result = parseWithThinking(chunk);

    assert.equal(result.thinkingText, expected);
    assert.equal(result.visibleText, "");
    assert.deepEqual(result.toolCalls, []);
  }
});

test("keeps partial Qwen3 plain answers visible when thinking is disabled", () => {
  const result = parse("I am Qwen3.5, a large language model ");

  assert.equal(result.thinkingText, "");
  assert.equal(result.visibleText, "I am Qwen3.5, a large language model");
  assert.deepEqual(result.toolCalls, []);
});

test("does not expose partial Qwen3 tool calls as visible text", () => {
  const chunks = ["<", "<tool", "<tool_call", "<tool_call>", "<tool_call>\n<function=getWeather>", "<tool_call>\n<function=getWeather>\n<parameter=location>", "<tool_call>\n<function=getWeather>\n<parameter=location>\nBern", "<tool_call>\n<function=getWeather>\n<parameter=location>\nBern\n</parameter>", "<tool_call>\n<function=getWeather>\n<parameter=location>\nBern\n</parameter>\n</function>", "<tool_call>\n<function=getWeather>\n<parameter=location>\nBern\n</parameter>\n</function>\n</tool_call"];

  for (const chunk of chunks) {
    const result = parse(chunk);

    assert.equal(result.visibleText, "");
    assert.deepEqual(result.toolCalls, []);
  }
});
