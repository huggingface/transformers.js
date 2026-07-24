import assert from "node:assert/strict";
import test from "node:test";
import { Agent } from "../src/Agent";
import { Tool } from "../src/Tool";
import { ModelAdapterBase } from "../src/adapters/ModelAdapterBase";
import type { Model } from "../src/Model";

test("returns tool calls without executing them and accepts an external response", async () => {
  const outputs = ['<think>Check the weather service.</think><tool_call>{"name":"get_weather","args":{"location":"London"}}</tool_call>', "It is sunny in London."];
  const conversations: Array<Array<Record<string, unknown>>> = [];
  let generateCount = 0;
  let executeCount = 0;

  const tokenizer = Object.assign(() => ({ input_ids: { dims: [1, 2], size: 2 } }), {
    apply_chat_template(conversation: Array<Record<string, unknown>>) {
      conversations.push(conversation);
      return "rendered prompt";
    },
    decode() {
      return outputs[generateCount - 1];
    },
  });
  const model = {
    modelId: "test-model",
    isInitialized: true,
    tokenizer,
    model: {
      config: {},
      async generate() {
        generateCount += 1;
        return {
          dims: [1, 3],
          slice: () => ({ data: [1] }),
        };
      },
    },
  } as unknown as Model;

  const weatherTool = new Tool<{ location: string }>({
    name: "get_weather",
    description: "Get current weather.",
    parameters: {
      location: Tool.string(),
    },
    execute: ({ location }) => {
      executeCount += 1;
      return [{ type: "object", value: { location, condition: "sunny" } }];
    },
  });
  const agent = new Agent({
    model,
    adapter: new ModelAdapterBase(),
    tools: [weatherTool],
    enableThinking: true,
  });

  const first = await agent.prompt("What is the weather in London?");
  assert.equal(generateCount, 1);
  assert.equal(executeCount, 0);
  assert.equal(first.thinking, "Check the weather service.");
  assert.equal(first.response, "");
  assert.deepEqual(first.toolCalls, [
    {
      callID: "toolcall_1",
      name: "get_weather",
      arguments: { location: "London" },
    },
  ]);

  const toolResult = await weatherTool.execute(first.toolCalls[0].arguments as { location: string });
  assert.equal(executeCount, 1);

  first.toolCalls[0].arguments.location = "Paris";
  const storedCall = agent.history[1].content;
  assert.equal(typeof storedCall === "string" ? undefined : storedCall[0].type, "tool-call");
  assert.deepEqual(typeof storedCall === "string" || storedCall[0].type !== "tool-call" ? undefined : storedCall[0].value.arguments, { location: "London" });

  await assert.rejects(
    agent.prompt([
      {
        role: "user",
        content: [
          {
            type: "tool-response",
            value: {
              callID: "unknown-call",
              name: "get_weather",
              result: [{ type: "text", value: "sunny" }],
            },
          },
        ],
      },
    ]),
    /Unknown tool call ID/,
  );
  assert.equal(generateCount, 1);
  assert.equal(agent.history.length, 2);

  const second = await agent.prompt([
    {
      role: "user",
      content: [
        {
          type: "tool-response",
          value: {
            callID: first.toolCalls[0].callID,
            name: "get_weather",
            result: toolResult,
          },
        },
      ],
    },
  ]);

  assert.equal(generateCount, 2);
  assert.equal(second.response, "It is sunny in London.");
  assert.equal(second.thinking, "");
  assert.deepEqual(second.toolCalls, []);
  assert.deepEqual(conversations[1].slice(-2), [
    {
      role: "assistant",
      content: undefined,
      tool_calls: [
        {
          id: "toolcall_1",
          type: "function",
          function: {
            name: "get_weather",
            arguments: '{"location":"London"}',
          },
        },
      ],
    },
    {
      role: "tool",
      content: '{"location":"London","condition":"sunny"}',
      tool_call_id: "toolcall_1",
      name: "get_weather",
    },
  ]);
});
