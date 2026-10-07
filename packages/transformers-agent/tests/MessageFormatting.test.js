"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var strict_1 = require("node:assert/strict");
var node_test_1 = require("node:test");
var messageFormatting_1 = require("../src/messageFormatting");
(0, node_test_1.default)("selects only model families with special input formatting", function () {
    strict_1.default.equal((0, messageFormatting_1.getModelFamily)("onnx-community/gemma-4-E2B-it-ONNX"), "gemma4");
    strict_1.default.equal((0, messageFormatting_1.getModelFamily)("Qwen/Qwen3-8B"), "qwen3");
    strict_1.default.equal((0, messageFormatting_1.getModelFamily)("onnx-community/LFM2-ONNX"), "default");
});
(0, node_test_1.default)("keeps Qwen tool arguments structured", function () {
    var messages = [
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
    var message = (0, messageFormatting_1.formatMessages)(messages, "qwen3")[0];
    strict_1.default.deepEqual(message.tool_calls[0].function.arguments, {
        location: "London",
    });
});
(0, node_test_1.default)("formats structured tool responses for Gemma4", function () {
    var messages = [
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
    strict_1.default.deepEqual((0, messageFormatting_1.formatMessages)(messages, "gemma4"), [
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
(0, node_test_1.default)("preserves text tool results for Gemma4", function () {
    var messages = [
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
    strict_1.default.deepEqual((0, messageFormatting_1.formatMessages)(messages, "gemma4"), [
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
(0, node_test_1.default)("formats failed tool responses for Gemma4", function () {
    var messages = [
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
    strict_1.default.deepEqual((0, messageFormatting_1.formatMessages)(messages, "gemma4"), [
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
