import { env, ModelRegistry, PreTrainedTokenizer, AutoConfig, pipeline } from "../../types/transformers.js";
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
await ModelRegistry.get_files("org/model", options);
await ModelRegistry.get_pipeline_files("text-generation", "org/model", options);
await ModelRegistry.get_model_files("org/model", options);
await ModelRegistry.get_tokenizer_files("org/model", options);
await ModelRegistry.get_processor_files("org/model", options);
await ModelRegistry.get_file_metadata("org/model", "config.json", options);
await ModelRegistry.get_available_dtypes("org/model", options);

// Literal objects deliberately exercise excess-property checks (not structural variable assignment).
// @ts-expect-error Discovery does not dispatch progress callbacks.
await ModelRegistry.get_files("org/model", { progress_callback: () => {} });
// @ts-expect-error Discovery does not dispatch progress callbacks.
await ModelRegistry.get_pipeline_files("text-generation", "org/model", { progress_callback: () => {} });
// @ts-expect-error Discovery does not dispatch progress callbacks.
await ModelRegistry.get_model_files("org/model", { progress_callback: () => {} });
// @ts-expect-error Discovery does not dispatch progress callbacks.
await ModelRegistry.get_available_dtypes("org/model", { progress_callback: () => {} });
// @ts-expect-error Discovery does not dispatch progress callbacks.
await ModelRegistry.get_tokenizer_files("org/model", { progress_callback: () => {} });
// @ts-expect-error Discovery does not dispatch progress callbacks.
await ModelRegistry.get_processor_files("org/model", { progress_callback: () => {} });
// @ts-expect-error Metadata probes do not dispatch progress callbacks.
await ModelRegistry.get_file_metadata("org/model", "config.json", { progress_callback: () => {} });
// @ts-expect-error Cache discovery does not dispatch progress callbacks.
await ModelRegistry.is_cached("org/model", { progress_callback: () => {} });
// @ts-expect-error Cache discovery does not dispatch progress callbacks.
await ModelRegistry.is_cached_files("org/model", { progress_callback: () => {} });
// @ts-expect-error Cache discovery does not dispatch progress callbacks.
await ModelRegistry.is_pipeline_cached("text-generation", "org/model", { progress_callback: () => {} });
// @ts-expect-error Cache discovery does not dispatch progress callbacks.
await ModelRegistry.is_pipeline_cached_files("text-generation", "org/model", { progress_callback: () => {} });
// @ts-expect-error Cache clearing does not dispatch progress callbacks.
await ModelRegistry.clear_cache("org/model", { progress_callback: () => {} });
// @ts-expect-error Cache clearing does not dispatch progress callbacks.
await ModelRegistry.clear_pipeline_cache("text-generation", "org/model", { progress_callback: () => {} });

// Ordinary loading APIs still accept download progress callbacks.
await PreTrainedTokenizer.from_pretrained("org/model", { progress_callback: () => {} });
await AutoConfig.from_pretrained("org/model", { progress_callback: () => {} });
await pipeline("text-generation", "org/model", { progress_callback: () => {} });
