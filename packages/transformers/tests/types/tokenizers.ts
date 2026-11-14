import type { PreTrainedTokenizer, BatchEncoding } from "../../src/tokenization_utils.js";
import type { Tensor } from "../../src/utils/tensor.js";
import type { ResponseParser } from "../../src/utils/chat_parsing/response_parser.js";

import type { Expect, Equal, ExpectError } from "./_base.ts";

declare const tokenizer: PreTrainedTokenizer;

const conversation = [{ role: "user", content: "Hello!" }] as const;
type IsAssignable<T, U> = T extends U ? true : false;

// Callable tokenizer defaults to tensors.
{
  const output = tokenizer("hello");
  type T0 = Expect<Equal<typeof output, BatchEncoding<Tensor>>>;
  type T1 = Expect<Equal<typeof output.input_ids, Tensor>>;
  type T2 = Expect<Equal<typeof output.attention_mask, Tensor>>;
}

// Single text + arrays.
{
  const output = tokenizer("hello", { return_tensor: false });
  type T0 = Expect<Equal<typeof output, BatchEncoding<number[]>>>;
  type T1 = Expect<Equal<typeof output.input_ids, number[]>>;
  type T2 = Expect<Equal<typeof output.attention_mask, number[]>>;
  type T3 = ExpectError<IsAssignable<typeof output.input_ids, Tensor>>;
}

// Batch text + arrays.
{
  const output = tokenizer(["hello", "world"], { return_tensor: false });
  type T0 = Expect<Equal<typeof output, BatchEncoding<number[][]>>>;
  type T1 = Expect<Equal<typeof output.input_ids, number[][]>>;
  type T2 = Expect<Equal<typeof output.attention_mask, number[][]>>;
  type T3 = ExpectError<IsAssignable<typeof output.input_ids, number[]>>;
}

// _call mirrors the callable signature.
{
  const output = tokenizer._call("hello", { return_tensor: true });
  type T0 = Expect<Equal<typeof output, BatchEncoding<Tensor>>>;
  type T1 = Expect<Equal<typeof output.input_ids, Tensor>>;
  type T2 = ExpectError<IsAssignable<typeof output.input_ids, number[]>>;
}

{
  const output = tokenizer._call(["hello", "world"], { return_tensor: false });
  type T0 = Expect<Equal<typeof output, BatchEncoding<number[][]>>>;
  type T1 = Expect<Equal<typeof output.input_ids, number[][]>>;
}

// apply_chat_template narrows by tokenize / return_dict / return_tensor.
{
  const output = tokenizer.apply_chat_template([...conversation], { tokenize: false });
  type T1 = Expect<Equal<typeof output, string>>;
  type T2 = ExpectError<IsAssignable<typeof output, Tensor>>;
}

{
  const output = tokenizer.apply_chat_template([...conversation], {
    return_tensor: true,
    return_dict: true,
  });
  type T0 = Expect<Equal<typeof output, BatchEncoding<Tensor>>>;
  type T1 = Expect<Equal<typeof output.input_ids, Tensor>>;
}

{
  const output = tokenizer.apply_chat_template([...conversation], {
    return_tensor: false,
    return_dict: true,
  });
  type T0 = Expect<Equal<typeof output, BatchEncoding<number[]>>>;
  type T1 = Expect<Equal<typeof output.input_ids, number[]>>;
}

{
  const output = tokenizer.apply_chat_template([...conversation], {
    return_tensor: true,
    return_dict: false,
  });
  type T1 = Expect<Equal<typeof output, Tensor>>;
}

{
  const output = tokenizer.apply_chat_template([...conversation], {
    return_tensor: false,
    return_dict: false,
  });
  type T1 = Expect<Equal<typeof output, number[]>>;
}

// Response parsing preserves batch shape and exposes an incremental parser.
{
  const output = tokenizer.parse_response("first", { prefix: "" });
  type T0 = Expect<Equal<typeof output, Record<string, any>>>;
}

{
  const output = tokenizer.parse_response([1, 2, 3], { prefix: [1, 2] });
  type T0 = Expect<Equal<typeof output, Record<string, any>>>;
}

{
  const output = tokenizer.parse_response(["first", "second"], { prefix: "" });
  type T0 = Expect<Equal<typeof output, Record<string, any>[]>>;
}

{
  const output = tokenizer.parse_response([[1, 2], [3]], { prefix: "" });
  type T0 = Expect<Equal<typeof output, Record<string, any>[]>>;
}

{
  const parser = tokenizer.get_response_parser({ prefix: "" });
  type T0 = Expect<Equal<typeof parser, ResponseParser>>;
  const events = parser.feed("text");
  type T1 = Expect<Equal<(typeof events)[number]["type"], "region_open" | "region_chunk" | "region_close">>;
  const [message] = parser.finalize();
  type T2 = Expect<Equal<typeof message, Record<string, any>>>;
}
