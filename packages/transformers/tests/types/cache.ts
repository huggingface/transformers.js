import { DynamicCache } from "../../src/cache_utils.js";

import type { Tensor } from "../../src/utils/tensor.js";


import type { Expect, Equal } from "./_base.ts";

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
