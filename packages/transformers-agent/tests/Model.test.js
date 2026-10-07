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
Object.defineProperty(exports, "__esModule", { value: true });
var strict_1 = require("node:assert/strict");
var node_test_1 = require("node:test");
var transformers_1 = require("@huggingface/transformers");
var Model_1 = require("../src/Model");
(0, node_test_1.default)("Model loads inference backends through the text-generation pipeline", function () { return __awaiter(void 0, void 0, void 0, function () {
    var backend, tokenizer, loadedModel, progressCallback, calls, model;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                backend = {
                    modelId: "test/custom-backend",
                    load: function () {
                        return __awaiter(this, void 0, void 0, function () {
                            return __generator(this, function (_a) {
                                throw new Error("The pipeline stub should handle loading.");
                            });
                        });
                    },
                };
                tokenizer = { chat_template: "backend chat template" };
                loadedModel = { config: { model_type: "test" } };
                progressCallback = function () { };
                calls = [];
                (0, transformers_1.__setPipelineImplementation)(function () {
                    var args = [];
                    for (var _i = 0; _i < arguments.length; _i++) {
                        args[_i] = arguments[_i];
                    }
                    return __awaiter(void 0, void 0, void 0, function () {
                        return __generator(this, function (_a) {
                            calls.push(args);
                            return [2 /*return*/, { tokenizer: tokenizer, model: loadedModel }];
                        });
                    });
                });
                model = new Model_1.Model({
                    modelId: backend,
                    device: "webgpu",
                    dtype: "auto",
                });
                strict_1.default.equal(model.modelId, backend.modelId);
                return [4 /*yield*/, model.init(progressCallback)];
            case 1:
                _a.sent();
                strict_1.default.equal(model.isInitialized, true);
                strict_1.default.equal(model.tokenizer, tokenizer);
                strict_1.default.equal(model.tokenizer.chat_template, "backend chat template");
                strict_1.default.equal(model.model, loadedModel);
                strict_1.default.deepEqual(calls, [
                    [
                        "text-generation",
                        backend,
                        {
                            revision: "main",
                            device: "webgpu",
                            dtype: "auto",
                            progress_callback: progressCallback,
                        },
                    ],
                ]);
                return [4 /*yield*/, model.init(progressCallback)];
            case 2:
                _a.sent();
                strict_1.default.equal(calls.length, 1);
                return [2 /*return*/];
        }
    });
}); });
(0, node_test_1.default)("forwards the configured revision to loading and cache operations", function () { return __awaiter(void 0, void 0, void 0, function () {
    var registry, pipelineCalls, model, _a, _b;
    return __generator(this, function (_c) {
        switch (_c.label) {
            case 0:
                registry = transformers_1.ModelRegistry;
                registry.calls.length = 0;
                pipelineCalls = [];
                (0, transformers_1.__setPipelineImplementation)(function () {
                    var args = [];
                    for (var _i = 0; _i < arguments.length; _i++) {
                        args[_i] = arguments[_i];
                    }
                    return __awaiter(void 0, void 0, void 0, function () {
                        return __generator(this, function (_a) {
                            pipelineCalls.push(args);
                            return [2 /*return*/, { tokenizer: {}, model: {} }];
                        });
                    });
                });
                model = new Model_1.Model({
                    modelId: "onnx-community/gemma-4-E2B-it-ONNX",
                    revision: "refs/pr/5",
                    device: "webgpu",
                    dtype: "q4f16",
                });
                return [4 /*yield*/, model.init()];
            case 1:
                _c.sent();
                return [4 /*yield*/, model.isCached()];
            case 2:
                _c.sent();
                _b = (_a = strict_1.default).equal;
                return [4 /*yield*/, model.downloadSize()];
            case 3:
                _b.apply(_a, [_c.sent(), 0]);
                strict_1.default.equal(model.revision, "refs/pr/5");
                strict_1.default.deepEqual(pipelineCalls, [
                    [
                        "text-generation",
                        "onnx-community/gemma-4-E2B-it-ONNX",
                        {
                            revision: "refs/pr/5",
                            device: "webgpu",
                            dtype: "q4f16",
                            progress_callback: undefined,
                        },
                    ],
                ]);
                strict_1.default.deepEqual(registry.calls, [
                    ["is_pipeline_cached", "text-generation", "onnx-community/gemma-4-E2B-it-ONNX", { revision: "refs/pr/5", device: "webgpu", dtype: "q4f16" }],
                    ["get_pipeline_files", "text-generation", "onnx-community/gemma-4-E2B-it-ONNX", { revision: "refs/pr/5", device: "webgpu", dtype: "q4f16" }],
                    ["get_file_metadata", "onnx-community/gemma-4-E2B-it-ONNX", "model.onnx", { revision: "refs/pr/5" }],
                ]);
                return [2 /*return*/];
        }
    });
}); });
(0, node_test_1.default)("defaults revision to main", function () {
    strict_1.default.equal(new Model_1.Model({ modelId: "test-model" }).revision, "main");
});
