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
  assert.equal(agent.getLatestUsage(), null);

  const first = await agent.prompt("What is the weather in London?");
  assert.equal(generateCount, 1);
  assert.equal(executeCount, 0);
  assert.deepEqual(first, [
    {
      type: "thinking",
      value: "Check the weather service.",
    },
    {
      type: "tool-call",
      value: {
        callID: "toolcall_1",
        name: "get_weather",
        arguments: { location: "London" },
      },
    },
  ]);
  const firstUsage = agent.getLatestUsage();
  assertUsage(firstUsage);
  if (!firstUsage) throw new Error("Expected usage.");
  firstUsage.totalTokens = 0;
  assertUsage(agent.getLatestUsage());
  const firstCall = first.find((part) => part.type === "tool-call");
  if (!firstCall) throw new Error("Expected a tool call.");

  const toolResult = await weatherTool.execute(firstCall.value.arguments as { location: string });
  assert.equal(executeCount, 1);

  firstCall.value.arguments.location = "Paris";
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
            callID: firstCall.value.callID,
            name: "get_weather",
            result: toolResult,
          },
        },
      ],
    },
  ]);

  assert.equal(generateCount, 2);
  assert.deepEqual(second, [{ type: "text", value: "It is sunny in London." }]);
  assertUsage(agent.getLatestUsage());
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
  agent.clearHistory();
  assert.equal(agent.getLatestUsage(), null);
});

test("streams incremental thinking and text content and closes without a done chunk", async () => {
  const tokenizer = Object.assign(() => ({ input_ids: { dims: [1, 2], size: 2 } }), {
    apply_chat_template() {
      return "rendered prompt";
    },
    decode() {
      return "<think>Check first.</think>Check that result.";
    },
  });
  const model = {
    modelId: "test-model",
    isInitialized: true,
    tokenizer,
    model: {
      config: {},
      async generate({ streamer }: { streamer: { callback_function: (text: string) => void } }) {
        streamer.callback_function("<think>Check");
        streamer.callback_function(" first.</think>");
        streamer.callback_function("Check");
        streamer.callback_function(" that");
        streamer.callback_function(" result.");
        return {
          dims: [1, 3],
          slice: () => ({ data: [1] }),
        };
      },
    },
  } as unknown as Model;
  const agent = new Agent({ model, adapter: new ModelAdapterBase() });

  const chunks = [];
  for await (const chunk of agent.promptStreaming("Check this")) chunks.push(chunk);

  assert.deepEqual(chunks, [
    { type: "thinking", value: "Check" },
    { type: "thinking", value: " first." },
    { type: "text", value: "Check" },
    { type: "text", value: " that" },
    { type: "text", value: " result." },
  ]);
  assertUsage(agent.getLatestUsage());
});

test("does not reconcile streamed deltas against a different final decode", async () => {
  const tokenizer = Object.assign(() => ({ input_ids: { dims: [1, 2], size: 2 } }), {
    apply_chat_template() {
      return "rendered prompt";
    },
    decode() {
      return "<think>Final thinking.</think>Final response.";
    },
  });
  const model = {
    modelId: "test-model",
    isInitialized: true,
    tokenizer,
    model: {
      config: {},
      async generate({ streamer }: { streamer: { callback_function: (text: string) => void } }) {
        streamer.callback_function("<think>Streamed thinking.</think>");
        streamer.callback_function("Streamed response.");
        return {
          dims: [1, 3],
          slice: () => ({ data: [1] }),
        };
      },
    },
  } as unknown as Model;
  const agent = new Agent({ model, adapter: new ModelAdapterBase() });

  const chunks = [];
  for await (const chunk of agent.promptStreaming("Check this")) chunks.push(chunk);

  assert.deepEqual(chunks, [
    { type: "thinking", value: "Streamed thinking." },
    { type: "text", value: "Streamed response." },
  ]);
  assertUsage(agent.getLatestUsage());
  assert.equal(agent.history.at(-1)?.content, "Final response.");
});

function assertUsage(usage: ReturnType<Agent["getLatestUsage"]>) {
  assert.ok(usage);
  assert.equal(usage.promptTokens, 2);
  assert.equal(usage.completionTokens, 1);
  assert.equal(usage.totalTokens, 3);
  assert.ok(Number.isFinite(usage.tokensPerSecond));
  assert.ok(usage.tokensPerSecond >= 0);
  assert.ok(usage.timeToFirstTokenMs >= 0);
  assert.ok(usage.generationTimeMs >= usage.timeToFirstTokenMs);
  assert.ok(usage.totalTimeMs >= usage.generationTimeMs);
}
