import { env, ModelRegistry, PreTrainedTokenizer } from "../../types/transformers.js";
import type { Expect, Equal } from "./_base.ts";

env.hfToken = null;
declare const tokenizer: PreTrainedTokenizer;
const batch: bigint[][] = [[1n], [2n]];
const result = tokenizer.parse_response(batch, { prefix: batch });
type BatchResult = Expect<Equal<typeof result, Record<string, any>[]>>;
const single = tokenizer.parse_response([1n], { prefix: batch });
type SingleResult = Expect<Equal<typeof single, Record<string, any>>>;
const options = { env: { hfToken: null, allowRemoteModels: false }, local_files_only: true };
await ModelRegistry.is_cached("org/model", options);
await ModelRegistry.is_pipeline_cached("text-generation", "org/model", options);
await ModelRegistry.clear_cache("org/model", options);
await ModelRegistry.clear_pipeline_cache("text-generation", "org/model", options);
