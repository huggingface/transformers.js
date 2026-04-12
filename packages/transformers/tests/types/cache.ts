import { DynamicCache } from "../../src/cache_utils.js";

import type { Tensor } from "../../src/utils/tensor.js";

import type { TextGenerationConfig } from "../../src/pipelines/text-generation.js";

import type { Expect, Equal, ExpectError } from "./_base.ts";

// Initialize Dynamic Cache
const cache = new DynamicCache();

// Ensure cache can be indexed with string keys and returns Tensors
type T1 = Expect<Equal<typeof cache['past_key_values.0.key'], Tensor>>;
type T2 = Expect<Equal<typeof cache['past_key_values.0.value'], Tensor>>;

// Ensure cache can be iterated over and entries are Tensors
for (const key in cache) {
    type T3 = Expect<Equal<typeof cache[typeof key], Tensor>>;
}

// Ensure tensors in cache can be disposed
type T4 = Expect<Equal<ReturnType<typeof cache.dispose>, Promise<void>>>;

// Ensure past_key_values accepts DynamicCache or null
type PastKeyValues = TextGenerationConfig['past_key_values'];
type T5 = Expect<Equal<PastKeyValues, DynamicCache | null | undefined>>;

// Ensure DynamicCache is assignable to past_key_values
type IsAssignable<T, U> = T extends U ? true : false;
type T6 = Expect<IsAssignable<DynamicCache, NonNullable<PastKeyValues>>>;
type T7 = Expect<IsAssignable<null, PastKeyValues>>;

// Ensure invalid types are NOT assignable to past_key_values
type T8 = ExpectError<IsAssignable<string, NonNullable<PastKeyValues>>>;
type T9 = ExpectError<IsAssignable<number, NonNullable<PastKeyValues>>>;
type T10 = ExpectError<IsAssignable<boolean, NonNullable<PastKeyValues>>>;
type T11 = ExpectError<IsAssignable<object, NonNullable<PastKeyValues>>>;
