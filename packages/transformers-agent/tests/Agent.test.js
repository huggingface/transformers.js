"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __asyncValues = (this && this.__asyncValues) || function (o) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var m = o[Symbol.asyncIterator], i;
    return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
    function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
    function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
};
Object.defineProperty(exports, "__esModule", { value: true });
var strict_1 = require("node:assert/strict");
var node_test_1 = require("node:test");
var Agent_1 = require("../src/Agent");
var Tool_1 = require("../src/Tool");
var ModelAdapterBase_1 = require("../src/adapters/ModelAdapterBase");
(0, node_test_1.default)("returns tool calls without executing them and accepts an external response", function () { return __awaiter(void 0, void 0, void 0, function () {
    var outputs, conversations, generateCount, executeCount, tokenizer, model, weatherTool, agent, first, firstCall, toolResult, storedCall, second;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                outputs = ['<think>Check the weather service.</think><tool_call>{"name":"get_weather","args":{"location":"London"}}</tool_call>', "It is sunny in London."];
                conversations = [];
                generateCount = 0;
                executeCount = 0;
                tokenizer = Object.assign(function () { return ({ input_ids: { dims: [1, 2], size: 2 } }); }, {
                    apply_chat_template: function (conversation) {
                        conversations.push(conversation);
                        return "rendered prompt";
                    },
                    decode: function () {
                        return outputs[generateCount - 1];
                    },
                });
                model = {
                    modelId: "test-model",
                    isInitialized: true,
                    tokenizer: tokenizer,
                    model: {
                        config: {},
                        generate: function () {
                            return __awaiter(this, void 0, void 0, function () {
                                return __generator(this, function (_a) {
                                    generateCount += 1;
                                    return [2 /*return*/, {
                                            dims: [1, 3],
                                            slice: function () { return ({ data: [1] }); },
                                        }];
                                });
                            });
                        },
                    },
                };
                weatherTool = new Tool_1.Tool({
                    name: "get_weather",
                    description: "Get current weather.",
                    parameters: {
                        location: Tool_1.Tool.string(),
                    },
                    execute: function (_a) {
                        var location = _a.location;
                        executeCount += 1;
                        return [{ type: "object", value: { location: location, condition: "sunny" } }];
                    },
                });
                agent = new Agent_1.Agent({
                    model: model,
                    adapter: new ModelAdapterBase_1.ModelAdapterBase(),
                    tools: [weatherTool],
                    enableThinking: true,
                });
                return [4 /*yield*/, agent.prompt("What is the weather in London?")];
            case 1:
                first = _a.sent();
                strict_1.default.equal(generateCount, 1);
                strict_1.default.equal(executeCount, 0);
                strict_1.default.deepEqual(first, [
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
                    {
                        type: "usage",
                        value: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
                    },
                ]);
                firstCall = first.find(function (part) { return part.type === "tool-call"; });
                if (!firstCall)
                    throw new Error("Expected a tool call.");
                return [4 /*yield*/, weatherTool.execute(firstCall.value.arguments)];
            case 2:
                toolResult = _a.sent();
                strict_1.default.equal(executeCount, 1);
                firstCall.value.arguments.location = "Paris";
                storedCall = agent.history[1].content;
                strict_1.default.equal(typeof storedCall === "string" ? undefined : storedCall[0].type, "tool-call");
                strict_1.default.deepEqual(typeof storedCall === "string" || storedCall[0].type !== "tool-call" ? undefined : storedCall[0].value.arguments, { location: "London" });
                return [4 /*yield*/, strict_1.default.rejects(agent.prompt([
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
                    ]), /Unknown tool call ID/)];
            case 3:
                _a.sent();
                strict_1.default.equal(generateCount, 1);
                strict_1.default.equal(agent.history.length, 2);
                return [4 /*yield*/, agent.prompt([
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
                    ])];
            case 4:
                second = _a.sent();
                strict_1.default.equal(generateCount, 2);
                strict_1.default.deepEqual(second, [
                    { type: "text", value: "It is sunny in London." },
                    {
                        type: "usage",
                        value: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
                    },
                ]);
                strict_1.default.deepEqual(conversations[1].slice(-2), [
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
                return [2 /*return*/];
        }
    });
}); });
(0, node_test_1.default)("streams incremental thinking and text content and closes without a done chunk", function () { return __awaiter(void 0, void 0, void 0, function () {
    var tokenizer, model, agent, chunks, _a, _b, _c, chunk, e_1_1;
    var _d, e_1, _e, _f;
    return __generator(this, function (_g) {
        switch (_g.label) {
            case 0:
                tokenizer = Object.assign(function () { return ({ input_ids: { dims: [1, 2], size: 2 } }); }, {
                    apply_chat_template: function () {
                        return "rendered prompt";
                    },
                    decode: function () {
                        return "<think>Check first.</think>Check that result.";
                    },
                });
                model = {
                    modelId: "test-model",
                    isInitialized: true,
                    tokenizer: tokenizer,
                    model: {
                        config: {},
                        generate: function (_a) {
                            return __awaiter(this, arguments, void 0, function (_b) {
                                var streamer = _b.streamer;
                                return __generator(this, function (_c) {
                                    streamer.callback_function("<think>Check");
                                    streamer.callback_function(" first.</think>");
                                    streamer.callback_function("Check");
                                    streamer.callback_function(" that");
                                    streamer.callback_function(" result.");
                                    return [2 /*return*/, {
                                            dims: [1, 3],
                                            slice: function () { return ({ data: [1] }); },
                                        }];
                                });
                            });
                        },
                    },
                };
                agent = new Agent_1.Agent({ model: model, adapter: new ModelAdapterBase_1.ModelAdapterBase() });
                chunks = [];
                _g.label = 1;
            case 1:
                _g.trys.push([1, 6, 7, 12]);
                _a = true, _b = __asyncValues(agent.promptStreaming("Check this"));
                _g.label = 2;
            case 2: return [4 /*yield*/, _b.next()];
            case 3:
                if (!(_c = _g.sent(), _d = _c.done, !_d)) return [3 /*break*/, 5];
                _f = _c.value;
                _a = false;
                chunk = _f;
                chunks.push(chunk);
                _g.label = 4;
            case 4:
                _a = true;
                return [3 /*break*/, 2];
            case 5: return [3 /*break*/, 12];
            case 6:
                e_1_1 = _g.sent();
                e_1 = { error: e_1_1 };
                return [3 /*break*/, 12];
            case 7:
                _g.trys.push([7, , 10, 11]);
                if (!(!_a && !_d && (_e = _b.return))) return [3 /*break*/, 9];
                return [4 /*yield*/, _e.call(_b)];
            case 8:
                _g.sent();
                _g.label = 9;
            case 9: return [3 /*break*/, 11];
            case 10:
                if (e_1) throw e_1.error;
                return [7 /*endfinally*/];
            case 11: return [7 /*endfinally*/];
            case 12:
                strict_1.default.deepEqual(chunks, [
                    { type: "thinking", value: "Check" },
                    { type: "thinking", value: " first." },
                    { type: "text", value: "Check" },
                    { type: "text", value: " that" },
                    { type: "text", value: " result." },
                    {
                        type: "usage",
                        value: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
                    },
                ]);
                return [2 /*return*/];
        }
    });
}); });
(0, node_test_1.default)("does not reconcile streamed deltas against a different final decode", function () { return __awaiter(void 0, void 0, void 0, function () {
    var tokenizer, model, agent, chunks, _a, _b, _c, chunk, e_2_1;
    var _d, e_2, _e, _f;
    var _g;
    return __generator(this, function (_h) {
        switch (_h.label) {
            case 0:
                tokenizer = Object.assign(function () { return ({ input_ids: { dims: [1, 2], size: 2 } }); }, {
                    apply_chat_template: function () {
                        return "rendered prompt";
                    },
                    decode: function () {
                        return "<think>Final thinking.</think>Final response.";
                    },
                });
                model = {
                    modelId: "test-model",
                    isInitialized: true,
                    tokenizer: tokenizer,
                    model: {
                        config: {},
                        generate: function (_a) {
                            return __awaiter(this, arguments, void 0, function (_b) {
                                var streamer = _b.streamer;
                                return __generator(this, function (_c) {
                                    streamer.callback_function("<think>Streamed thinking.</think>");
                                    streamer.callback_function("Streamed response.");
                                    return [2 /*return*/, {
                                            dims: [1, 3],
                                            slice: function () { return ({ data: [1] }); },
                                        }];
                                });
                            });
                        },
                    },
                };
                agent = new Agent_1.Agent({ model: model, adapter: new ModelAdapterBase_1.ModelAdapterBase() });
                chunks = [];
                _h.label = 1;
            case 1:
                _h.trys.push([1, 6, 7, 12]);
                _a = true, _b = __asyncValues(agent.promptStreaming("Check this"));
                _h.label = 2;
            case 2: return [4 /*yield*/, _b.next()];
            case 3:
                if (!(_c = _h.sent(), _d = _c.done, !_d)) return [3 /*break*/, 5];
                _f = _c.value;
                _a = false;
                chunk = _f;
                chunks.push(chunk);
                _h.label = 4;
            case 4:
                _a = true;
                return [3 /*break*/, 2];
            case 5: return [3 /*break*/, 12];
            case 6:
                e_2_1 = _h.sent();
                e_2 = { error: e_2_1 };
                return [3 /*break*/, 12];
            case 7:
                _h.trys.push([7, , 10, 11]);
                if (!(!_a && !_d && (_e = _b.return))) return [3 /*break*/, 9];
                return [4 /*yield*/, _e.call(_b)];
            case 8:
                _h.sent();
                _h.label = 9;
            case 9: return [3 /*break*/, 11];
            case 10:
                if (e_2) throw e_2.error;
                return [7 /*endfinally*/];
            case 11: return [7 /*endfinally*/];
            case 12:
                strict_1.default.deepEqual(chunks, [
                    { type: "thinking", value: "Streamed thinking." },
                    { type: "text", value: "Streamed response." },
                    {
                        type: "usage",
                        value: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
                    },
                ]);
                strict_1.default.equal((_g = agent.history.at(-1)) === null || _g === void 0 ? void 0 : _g.content, "Final response.");
                return [2 /*return*/];
        }
    });
}); });
