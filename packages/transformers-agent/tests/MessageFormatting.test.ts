import assert from "node:assert/strict";
import test from "node:test";
import { formatMessages, getModelFamily } from "../src/messageFormatting";
import type { Message } from "../src/types";

test("selects only model families with special input formatting", () => {
  assert.equal(getModelFamily("onnx-community/gemma-4-E2B-it-ONNX"), "gemma4");
  assert.equal(getModelFamily("Qwen/Qwen3-8B"), "qwen3");
  assert.equal(getModelFamily("onnx-community/LFM2-ONNX"), "default");
});

test("keeps Qwen tool arguments structured", () => {
  const messages: Message[] = [
    {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          value: { callID: "toolcall_1", name: "get_weather", arguments: { location: "London" } },
        },
      ],
    },
  ];

  const [message] = formatMessages(messages, "qwen3");
  assert.deepEqual((message.tool_calls as Array<{ function: { arguments: unknown } }>)[0].function.arguments, {
    location: "London",
  });
});

test("formats structured tool responses for Gemma4, including turns with reasoning", () => {
  const messages: Array<Message & { thinking?: string }> = [
    {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          value: {
            callID: "toolcall_1",
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool-response",
          value: {
            callID: "toolcall_1",
            name: "get_weather",
            result: [{ type: "object", value: { location: "London", temperature: 20, weather: "sunny" } }],
          },
        },
      ],
    },
  ];

  assert.deepEqual(formatMessages(messages, "gemma4"), [
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

  messages[0].thinking = "Check the weather service.";
  const [withReasoning] = formatMessages(messages, "gemma4");
  assert.equal(withReasoning.reasoning_content, "Check the weather service.");
  assert.equal(withReasoning.content, undefined);
  assert.deepEqual(withReasoning.tool_calls, [
    {
      function: { name: "get_weather", arguments: { location: "London" } },
    },
  ]);
  assert.deepEqual(withReasoning.tool_responses, [
    {
      name: "get_weather",
      response: { location: "London", temperature: 20, weather: "sunny" },
    },
  ]);
});

test("preserves text tool results for Gemma4", () => {
  const messages: Message[] = [
    {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          value: {
            callID: "toolcall_1",
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool-response",
          value: {
            callID: "toolcall_1",
            name: "get_weather",
            result: [{ type: "text", value: "The weather in London in Sunny, 20 degrees celsius." }],
          },
        },
      ],
    },
  ];

  assert.deepEqual(formatMessages(messages, "gemma4"), [
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
          response: "The weather in London in Sunny, 20 degrees celsius.",
        },
      ],
    },
  ]);
});

test("formats failed tool responses for Gemma4", () => {
  const messages: Message[] = [
    {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          value: {
            callID: "toolcall_1",
            name: "get_weather",
            arguments: { location: "London" },
          },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool-response",
          value: {
            callID: "toolcall_1",
            name: "get_weather",
            errorMessage: "Weather service unavailable",
          },
        },
      ],
    },
  ];

  assert.deepEqual(formatMessages(messages, "gemma4"), [
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
          response: { error: "Weather service unavailable" },
        },
      ],
    },
  ]);
});
