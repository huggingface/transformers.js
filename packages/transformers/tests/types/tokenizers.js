"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
var conversation = [{ role: "user", content: "Hello!" }];
// Callable tokenizer defaults to tensors.
{
    var output = tokenizer("hello");
}
// Single text + arrays.
{
    var output = tokenizer("hello", { return_tensor: false });
}
// Batch text + arrays.
{
    var output = tokenizer(["hello", "world"], { return_tensor: false });
}
// _call mirrors the callable signature.
{
    var output = tokenizer._call("hello", { return_tensor: true });
}
{
    var output = tokenizer._call(["hello", "world"], { return_tensor: false });
}
// apply_chat_template narrows by tokenize / return_dict / return_tensor.
{
    var output = tokenizer.apply_chat_template(__spreadArray([], conversation, true), { tokenize: false });
}
{
    var output = tokenizer.apply_chat_template(__spreadArray([], conversation, true), {
        return_tensor: true,
        return_dict: true,
    });
}
{
    var output = tokenizer.apply_chat_template(__spreadArray([], conversation, true), {
        return_tensor: false,
        return_dict: true,
    });
}
{
    var output = tokenizer.apply_chat_template(__spreadArray([], conversation, true), {
        return_tensor: true,
        return_dict: false,
    });
}
{
    var output = tokenizer.apply_chat_template(__spreadArray([], conversation, true), {
        return_tensor: false,
        return_dict: false,
    });
}
// Response parsing preserves batch shape and exposes an incremental parser.
{
    var output = tokenizer.parse_response("first", { prefix: "" });
}
{
    var output = tokenizer.parse_response([1, 2, 3], { prefix: [1, 2] });
}
{
    var output = tokenizer.parse_response(["first", "second"], { prefix: "" });
}
{
    var output = tokenizer.parse_response([[1, 2], [3]], { prefix: "" });
}
{
    var parser = tokenizer.get_response_parser({ prefix: "" });
    var events = parser.feed("text");
    var message = parser.finalize()[0];
}
