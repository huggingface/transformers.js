import assert from "node:assert/strict";
import test from "node:test";
import { ModelAdapterBase } from "../src/adapters/ModelAdapterBase";

test("keeps incomplete thinking and tool calls out of visible streaming text", () => {
  const adapter = new ModelAdapterBase();
  const nextId = () => "toolcall_1";

  assert.deepEqual(adapter.parseAssistantContent("<think>Checking weather", nextId), {
    thinkingText: "Checking weather",
    visibleText: "",
    toolCalls: [],
  });
  assert.deepEqual(adapter.parseAssistantContent('Answer<tool_call>{"name":"get_weather"', nextId), {
    thinkingText: "",
    visibleText: "Answer",
    toolCalls: [],
  });
});
