import type {
  PreTrainedTokenizer,
  BatchEncoding,
} from "../../src/tokenization_utils.js";
import type { Tensor } from "../../src/utils/tensor.js";

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
