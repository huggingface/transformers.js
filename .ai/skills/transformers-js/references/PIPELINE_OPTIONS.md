# Pipeline options

The `pipeline(task, model?, options?)` factory accepts a common set of options
across every task. Task-specific call options are documented on each pipeline
class in the [API reference](https://huggingface.co/docs/transformers.js/api/pipelines).

## Common options

<!-- @generated:start id=typedef:PretrainedModelOptions -->
| Option | Type | Description |
|--------|------|-------------|
| `progress_callback`? | `ProgressCallback` | If specified, this function will be called during model construction, to provide the user with progress updates. _(default: `null`)_ |
| `config`? | `PretrainedConfig` | Configuration for the model to use instead of an automatically loaded configuration. Configuration can be automatically loaded when: - The model is a model provided by the library (loaded with the *model id* string of a pretrained model). - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a configuration JSON file named *config.json* is found in the directory. _(default: `null`)_ |
| `cache_dir`? | `string` | Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used. _(default: `null`)_ |
| `local_files_only`? | `boolean` | Whether or not to only look at local files (e.g., not try downloading the model). _(default: `false`)_ |
| `revision`? | `string` | The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. NOTE: This setting is ignored for local requests. _(default: `'main'`)_ |
| `subfolder`? | `string` | In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here. _(default: `'onnx'`)_ |
| `model_file_name`? | `string` | If specified, load the model with this name (excluding the dtype and .onnx suffixes). Currently only valid for encoder- or decoder-only models. _(default: `null`)_ |
| `device`? | `DeviceType|Record<string, DeviceType>` | The device to run the model on. If not specified, the device will be chosen from the environment settings. _(default: `null`)_ |
| `dtype`? | `DataType|Record<string, DataType>` | The data type to use for the model. If not specified, the data type will be chosen from the environment settings. _(default: `null`)_ |
| `use_external_data_format`? | `ExternalData|Record<string, ExternalData>` | Whether to load the model using the external data format (used for models >= 2GB in size). _(default: `false`)_ |
| `session_options`? | `InferenceSession.SessionOptions` | (Optional) User-specified session options passed to the runtime. If not provided, suitable defaults will be chosen. |
<!-- @generated:end id=typedef:PretrainedModelOptions -->

## Progress tracking

```javascript
const pipe = await pipeline("sentiment-analysis", null, {
  progress_callback: (info) => {
    if (info.status === "progress") {
      console.log(`${info.file}: ${info.progress.toFixed(1)}%`);
    }
  },
});
```

### `ProgressInfo` shape

<!-- @generated:start id=typedef:ProgressInfo -->
**`InitiateProgressInfo`**

| Option | Type | Description |
|--------|------|-------------|
| `status` | `'initiate'` |  |
| `name` | `string` | The model id or directory path. |
| `file` | `string` | The name of the file. |

**`DownloadProgressInfo`**

| Option | Type | Description |
|--------|------|-------------|
| `status` | `'download'` |  |
| `name` | `string` | The model id or directory path. |
| `file` | `string` | The name of the file. |

**`ProgressStatusInfo`**

| Option | Type | Description |
|--------|------|-------------|
| `status` | `'progress'` |  |
| `name` | `string` | The model id or directory path. |
| `file` | `string` | The name of the file. |
| `progress` | `number` | A number between 0 and 100. |
| `loaded` | `number` | The number of bytes loaded. |
| `total` | `number` | The total number of bytes to be loaded. |

**`DoneProgressInfo`**

| Option | Type | Description |
|--------|------|-------------|
| `status` | `'done'` |  |
| `name` | `string` | The model id or directory path. |
| `file` | `string` | The name of the file. |

**`ReadyProgressInfo`**

| Option | Type | Description |
|--------|------|-------------|
| `status` | `'ready'` |  |
| `task` | `string` | The loaded task. |
| `model` | `string` | The loaded model. |

**`TotalProgressInfo`**

| Option | Type | Description |
|--------|------|-------------|
| `status` | `'progress_total'` |  |
| `name` | `string` | The model id or directory path. |
| `progress` | `number` | A number between 0 and 100. |
| `loaded` | `number` | The number of bytes loaded. |
| `total` | `number` | The total number of bytes to be loaded. |
| `files` | `FilesLoadingMap` | A mapping of file names to their loading progress. |
<!-- @generated:end id=typedef:ProgressInfo -->

## Device selection

```javascript
// Default — CPU via WASM (most compatible)
await pipeline("sentiment-analysis", null);

// GPU via WebGPU (Chrome 113+, fastest for big models)
await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX", {
  device: "webgpu",
  dtype: "fp16",
});

// Node.js with onnxruntime-node
await pipeline("sentiment-analysis", null, { device: "cpu" });
```

Detect WebGPU support before attempting to use it:

```javascript
const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;
const pipe = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX", {
  device: hasWebGPU ? "webgpu" : "wasm",
  dtype: hasWebGPU ? "fp16" : "q8",
});
```

## Task-specific call options

Once created, each pipeline is itself callable with task-specific options:

```javascript
const classifier = await pipeline("text-classification");
await classifier("great movie", { top_k: 3 });

const generator = await pipeline("text-generation");
await generator("Once upon a time", { max_new_tokens: 100, temperature: 0.7 });

const translator = await pipeline("translation", "Xenova/nllb-200-distilled-600M");
await translator("Hello", { src_lang: "eng_Latn", tgt_lang: "fra_Latn" });
```

Task-specific recipes and every call option for each pipeline are in
[`TASKS.md`](./TASKS.md).

## Generation parameters

Every field below can be passed as an option to a `text-generation`,
`text2text-generation`, `translation`, or `summarization` pipeline call, or
anywhere `generate()` is invoked directly.

<!-- @generated:start id=fields:GenerationConfig -->
| Option | Type | Description |
|--------|------|-------------|
| `max_length`? | `number` | The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set. _(default: `20`)_ |
| `max_new_tokens`? | `number` | The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. _(default: `null`)_ |
| `min_length`? | `number` | The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set. _(default: `0`)_ |
| `min_new_tokens`? | `number` | The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt. _(default: `null`)_ |
| `early_stopping`? | `boolean|"never"` | Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: - `true`, where the generation stops as soon as there are `num_beams` complete candidates; - `false`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; - `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm). _(default: `false`)_ |
| `max_time`? | `number` | The maximum amount of time you allow the computation to run for in seconds. Generation will still finish the current pass after allocated time has been passed. _(default: `null`)_ |
| `do_sample`? | `boolean` | Whether or not to use sampling; use greedy decoding otherwise. _(default: `false`)_ |
| `num_beams`? | `number` | Number of beams for beam search. 1 means no beam search. _(default: `1`)_ |
| `num_beam_groups`? | `number` | Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. See [this paper](https://huggingface.co/papers/1610.02424) for more details. _(default: `1`)_ |
| `penalty_alpha`? | `number` | The values balance the model confidence and the degeneration penalty in contrastive search decoding. _(default: `null`)_ |
| `use_cache`? | `boolean` | Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding. _(default: `true`)_ |
| `temperature`? | `number` | The value used to modulate the next token probabilities. _(default: `1.0`)_ |
| `top_k`? | `number` | The number of highest probability vocabulary tokens to keep for top-k-filtering. _(default: `50`)_ |
| `top_p`? | `number` | If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. _(default: `1.0`)_ |
| `typical_p`? | `number` | Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this paper](https://huggingface.co/papers/2202.00666) for more details. _(default: `1.0`)_ |
| `epsilon_cutoff`? | `number` | If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more details. _(default: `0.0`)_ |
| `eta_cutoff`? | `number` | Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more details. _(default: `0.0`)_ |
| `diversity_penalty`? | `number` | This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled. _(default: `0.0`)_ |
| `repetition_penalty`? | `number` | The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://huggingface.co/papers/1909.05858) for more details. _(default: `1.0`)_ |
| `encoder_repetition_penalty`? | `number` | The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty. _(default: `1.0`)_ |
| `length_penalty`? | `number` | Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences. _(default: `1.0`)_ |
| `no_repeat_ngram_size`? | `number` | If set to int > 0, all ngrams of that size can only occur once. _(default: `0`)_ |
| `bad_words_ids`? | `number[][]` | List of token ids that are not allowed to be generated. In order to get the token ids of the words that should not appear in the generated text, use `tokenizer(bad_words, { add_prefix_space: true, add_special_tokens: false }).input_ids`. _(default: `null`)_ |
| `force_words_ids`? | `number[][]|number[][][]` | List of token ids that must be generated. If given a `number[][]`, this is treated as a simple list of words that must be included, the opposite to `bad_words_ids`. If given `number[][][]`, this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one can allow different forms of each word. _(default: `null`)_ |
| `renormalize_logits`? | `boolean` | Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `true` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization. _(default: `false`)_ |
| `constraints`? | `Object[]` | Custom constraints that can be added to the generation to ensure that the output will contain the use of certain tokens as defined by `Constraint` objects, in the most sensible way possible. _(default: `null`)_ |
| `forced_bos_token_id`? | `number` | The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for multilingual models like mBART where the first generated token needs to be the target language token. _(default: `null`)_ |
| `forced_eos_token_id`? | `number|number[]` | The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple *end-of-sequence* tokens. _(default: `null`)_ |
| `remove_invalid_values`? | `boolean` | Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation. _(default: `false`)_ |
| `exponential_decay_length_penalty`? | `[number, number]` | This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay. _(default: `null`)_ |
| `suppress_tokens`? | `number[]` | A list of tokens that will be suppressed at generation. The `SuppressTokens` logit processor will set their log probs to `-inf` so that they are not sampled. _(default: `null`)_ |
| `streamer`? | `TextStreamer` | A streamer that will be used to stream the generation. _(default: `null`)_ |
| `begin_suppress_tokens`? | `number[]` | A list of tokens that will be suppressed at the beginning of the generation. The `SuppressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled. _(default: `null`)_ |
| `forced_decoder_ids`? | `[number, number][]` | A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token of index 123. _(default: `null`)_ |
| `guidance_scale`? | `number` | The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality. _(default: `null`)_ |
| `num_return_sequences`? | `number` | The number of independently computed returned sequences for each element in the batch. _(default: `1`)_ |
| `output_attentions`? | `boolean` | Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more details. _(default: `false`)_ |
| `output_hidden_states`? | `boolean` | Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more details. _(default: `false`)_ |
| `output_scores`? | `boolean` | Whether or not to return the prediction scores. See `scores` under returned tensors for more details. _(default: `false`)_ |
| `return_dict_in_generate`? | `boolean` | Whether or not to return a `ModelOutput` instead of a plain tuple. _(default: `false`)_ |
| `pad_token_id`? | `number` | The id of the *padding* token. _(default: `null`)_ |
| `bos_token_id`? | `number` | The id of the *beginning-of-sequence* token. _(default: `null`)_ |
| `eos_token_id`? | `number|number[]` | The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens. _(default: `null`)_ |
| `encoder_no_repeat_ngram_size`? | `number` | If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`. _(default: `0`)_ |
| `decoder_start_token_id`? | `number` | If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token. _(default: `null`)_ |
| `generation_kwargs`? | `Object` | Additional generation kwargs will be forwarded to the `generate` function of the model. Kwargs that are not present in `generate`'s signature will be used in the model forward pass. _(default: ``)_ |
<!-- @generated:end id=fields:GenerationConfig -->

### Streaming tokens as they're produced

```javascript
import { pipeline, TextStreamer } from "@huggingface/transformers";

const generator = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX");
const streamer = new TextStreamer(generator.tokenizer, {
  skip_prompt: true,
  skip_special_tokens: true,
  callback_function: (text) => process.stdout.write(text),
});

await generator("Tell me a short story.", { max_new_tokens: 200, streamer });
```

For whisper-style chunk / token / finalize callbacks, use `WhisperTextStreamer`.

### KV cache reuse across calls

Passing the cache from one call to the next skips re-encoding the prefix:

```javascript
import { pipeline, DynamicCache } from "@huggingface/transformers";

const generator = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX");
const cache = new DynamicCache();
await generator("Hello,", { max_new_tokens: 20, past_key_values: cache });
await generator(" how are you?", { max_new_tokens: 20, past_key_values: cache });
```
