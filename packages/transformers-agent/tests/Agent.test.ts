import assert from "node:assert/strict";
import test from "node:test";
import { StructuredOutputProcessor } from "@huggingface/transformers-structured-output";
import { Agent } from "../src/Agent";
import { Tool } from "../src/Tool";
import type { Model } from "../src/Model";

test("returns tool calls without executing them and accepts an external response", async () => {
  const outputs = ['<think>Check the weather service.</think><tool_call>{"name":"get_weather","args":{"location":"London"}}</tool_call>', "It is sunny in London."];
  const conversations: Array<Array<Record<string, unknown>>> = [];
  const parseRequests: Array<{ prefix: string; tools: unknown[] }> = [];
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
    parse_response(response: string, options: { prefix: string; tools: unknown[] }) {
      parseRequests.push(options);
      return response.startsWith("<think>")
        ? {
            role: "assistant",
            thinking: "Check the weather service.",
            tool_calls: [
              {
                type: "function",
                function: { name: "get_weather", arguments: { location: "London" } },
              },
            ],
          }
        : { role: "assistant", content: response };
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
    tools: [weatherTool],
    enableThinking: true,
  });
  assert.equal(agent.getLatestUsage(), null);

  const first = await agent.prompt("What is the weather in London?");
  assert.equal(generateCount, 1);
  assert.equal(executeCount, 0);
  assert.equal(parseRequests[0].prefix, "rendered prompt");
  assert.deepEqual(parseRequests[0].tools, [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get current weather.",
        parameters: {
          type: "object",
          properties: { location: { type: "string" } },
          required: ["location"],
          additionalProperties: false,
        },
      },
    },
  ]);
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

test("applies Prompt API response constraints per turn across a tool follow-up", async () => {
  const outputs = ["tool", '{"answer":9}', "plain"];
  const conversations: Array<Array<Record<string, unknown>>> = [];
  const generateOptions: Array<Record<string, unknown>> = [];
  let generateCount = 0;
  const tokenizer = Object.assign(() => ({ input_ids: { dims: [1, 2], size: 2 } }), {
    apply_chat_template(conversation: Array<Record<string, unknown>>) {
      conversations.push(conversation);
      return "rendered prompt";
    },
    decode() {
      return outputs[generateCount - 1];
    },
    parse_response(response: string) {
      if (response === "tool") {
        return {
          role: "assistant",
          thinking: "Use the calculator.",
          tool_calls: [{ type: "function", function: { name: "add", arguments: { a: 4, b: 5 } } }],
        };
      }
      return { role: "assistant", content: response };
    },
  });
  const model = {
    modelId: "test-model",
    isInitialized: true,
    tokenizer,
    model: {
      config: {},
      async generate(options: Record<string, unknown>) {
        generateOptions.push(options);
        generateCount += 1;
        return { dims: [1, 3], slice: () => ({ data: [1] }) };
      },
    },
  } as unknown as Model;
  const agent = new Agent({
    model,
    enableThinking: true,
    initialPrompts: [{ role: "system", content: "You are a calculator." }],
  });
  const Processor = StructuredOutputProcessor as unknown as {
    instances: Array<{
      tokenizer: unknown;
      responseFormat: unknown;
      contexts: unknown[];
    }>;
  };
  const processorStart = Processor.instances.length;

  const first = await agent.prompt("What is 4+5?", {
    responseConstraint: /The answer is \d+\./,
  });
  const call = first.find((part) => part.type === "tool-call");
  if (!call) throw new Error("Expected a tool call.");

  const schema = {
    type: "object" as const,
    properties: { answer: { type: "integer" } },
    required: ["answer"],
    additionalProperties: false,
  };
  const second = await agent.prompt(
    [
      {
        role: "user",
        content: [
          {
            type: "tool-response",
            value: {
              callID: call.value.callID,
              name: call.value.name,
              result: [{ type: "object", value: { answer: 9 } }],
            },
          },
        ],
      },
    ],
    { responseConstraint: schema, omitResponseConstraintInput: true },
  );
  const third = await agent.prompt("Answer normally.");

  const processors = Processor.instances.slice(processorStart);
  assert.equal(processors.length, 2);
  assert.notEqual(processors[0], processors[1]);
  assert.equal(processors[0].tokenizer, tokenizer);
  assert.deepEqual(processors[0].responseFormat, { type: "regex", regex: "The answer is \\d+\\." });
  assert.deepEqual(processors[1].responseFormat, { type: "json_schema", json_schema: schema });
  assert.deepEqual(
    processors.map((processor) => processor.contexts),
    [[{ enable_thinking: true }], [{ enable_thinking: true }]],
  );
  assert.equal(generateOptions[0].logits_processor, processors[0]);
  assert.equal(generateOptions[1].logits_processor, processors[1]);
  assert.equal("logits_processor" in generateOptions[2], false);
  assert.equal(conversations[0][0].role, "system");
  assert.match(String(conversations[0][0].content), /^You are a calculator\./);
  assert.match(String(conversations[0][0].content), /regular expression: The answer is \\d\+\\\./);
  assert.equal(conversations[0][1].role, "user");
  assert.equal(
    conversations[1].some((message) => String(message.content).includes("Respond ONLY")),
    false,
  );
  assert.equal(
    conversations[2].some((message) => String(message.content).includes("Respond ONLY")),
    false,
  );
  assert.deepEqual(second, [{ type: "text", value: '{"answer":9}' }]);
  assert.deepEqual(third, [{ type: "text", value: "plain" }]);
  assert.equal(
    agent.history.some((message) => typeof message.content === "string" && message.content.includes("Respond ONLY")),
    false,
  );
  await assert.rejects(agent.prompt("Invalid options", { omitResponseConstraintInput: true }), /requires responseConstraint/);
  assert.throws(() => agent.promptStreaming("Invalid options", { omitResponseConstraintInput: true }), /requires responseConstraint/);
});

test("streams incremental thinking and text content and closes without a done chunk", async () => {
  const tokenizer = Object.assign(() => ({ input_ids: { dims: [1, 2], size: 2 } }), {
    apply_chat_template() {
      return "rendered prompt";
    },
    decode() {
      return "<think>Check first.</think>Check that result.";
    },
    parse_response() {
      return { role: "assistant", thinking: "Check first.", content: "Check that result." };
    },
    get_response_parser() {
      return createResponseParser((text) => {
        if (text === "<think>Check") return [chunk("thinking", "Check")];
        if (text === " first.</think>") return [chunk("thinking", " first.")];
        return [chunk("content", text)];
      });
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
  const agent = new Agent({ model });
  const Processor = StructuredOutputProcessor as unknown as { instances: unknown[] };
  const processorStart = Processor.instances.length;

  const chunks = [];
  for await (const chunk of agent.promptStreaming("Check this", { responseConstraint: /Check that result\./, omitResponseConstraintInput: true })) chunks.push(chunk);

  assert.deepEqual(chunks, [
    { type: "thinking", value: "Check" },
    { type: "thinking", value: " first." },
    { type: "text", value: "Check" },
    { type: "text", value: " that" },
    { type: "text", value: " result." },
  ]);
  assert.equal(Processor.instances.length, processorStart + 1);
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
    parse_response() {
      return { role: "assistant", thinking: "Final thinking.", content: "Final response." };
    },
    get_response_parser() {
      return createResponseParser(
        (text) => {
          if (text === "<think>Streamed thinking.</think>") {
            return [chunk("thinking", "Streamed thinking.")];
          }
          return [chunk("content", text, true)];
        },
        [{ type: "region_close", field: "content", value: "Streamed response." }],
      );
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
  const agent = new Agent({ model });

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

function chunk(field: string, text: string, dirty = false) {
  return { type: "region_chunk" as const, field, text, dirty };
}

function createResponseParser(feed: (text: string) => ReturnType<typeof chunk>[], finalEvents: Array<{ type: "region_close"; field: string; value: unknown }> = []) {
  return {
    initial_events: [],
    feed,
    finalize: () => [{ role: "assistant" }, finalEvents],
  };
}
