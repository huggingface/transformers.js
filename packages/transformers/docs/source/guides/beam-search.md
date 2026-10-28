# Beam search

Set `num_beams` above 1 in `generate()` to keep several candidate continuations
for each input. Beam search works with encoder-decoder and decoder-only models.

```javascript
import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@huggingface/transformers';

const model_id = 'Xenova/t5-small';
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const model = await AutoModelForSeq2SeqLM.from_pretrained(model_id);
const inputs = tokenizer('translate English to German: The house is wonderful.');

const sequences = await model.generate({
    ...inputs,
    num_beams: 4,
    num_return_sequences: 2,
    max_new_tokens: 30,
    length_penalty: 1.0,
    early_stopping: true,
});
console.log(tokenizer.batch_decode(sequences, { skip_special_tokens: true }));
```

`num_return_sequences` must be at most `num_beams`. Batched results are ordered
by input, then by descending hypothesis score. Use left padding for batched
decoder-only inputs.

Completed hypotheses are ranked by cumulative token log probability divided by
generated length raised to `length_penalty`. Prompt tokens are excluded from
this length. Positive penalties favor longer continuations; negative penalties
favor shorter ones. `early_stopping: true` stops once enough candidates finish,
`false` uses a score heuristic, and `"never"` uses a bound on the best attainable
score up to `max_length` or `max_new_tokens`.

## Diverse beam search

Split beams into groups and penalize tokens selected by earlier groups:

```javascript
const sequences = await model.generate({
    ...inputs,
    num_beams: 4,
    num_beam_groups: 2,
    diversity_penalty: 0.5,
    max_new_tokens: 30,
});
```

`num_beams` must be divisible by `num_beam_groups`. Each group searches
independently, with the diversity penalty included in its scores.

## Beam sampling

Set `do_sample: true` to sample candidates from the joint beam/token
distribution. Candidate pairs are sampled without replacement, then ranked by
their cumulative scores. `top_k` limits candidates per beam, with enough tokens
retained to seed the active beams and handle EOS candidates; `temperature`
controls sampling sharpness.

```javascript
const sequences = await model.generate({
    ...inputs,
    num_beams: 4,
    do_sample: true,
    top_k: 20,
    temperature: 0.8,
    max_new_tokens: 30,
});
```

## WebGPU

Load a model with `device: 'webgpu'` to run inference on WebGPU. Beam reordering
keeps aligned decoder-cache tensors on the GPU using buffer row copies. Candidate
ranking still runs in JavaScript. Tensors whose row sizes are not multiples of
four bytes use CPU row selection instead.

```javascript
const model = await AutoModelForSeq2SeqLM.from_pretrained(model_id, {
    device: 'webgpu',
    dtype: 'fp32',
});
```

## Restrictions

Beam search rejects streamers and classifier-free guidance (`guidance_scale > 1`).
Decoder-only multimodal inputs are not supported with beams.
Whisper token timestamps support short-form batches and multiple returned
hypotheses. Long-form seek requires one input and one returned sequence.
Timestamp alignment follows each hypothesis's beam ancestry and includes the
last prompt query, which predicts the first generated token. Normalization uses
the original prompt and generated attention rows before cropping all but the
final prompt row.

With `output_attentions: true`, attention tensors contain raw per-step beam rows;
`beam_indices` identifies the rows used for each selected hypothesis.
`sequence_lengths` excludes right padding. Internally retained attentions are
released after timestamp extraction unless explicitly requested by the caller.
Attention history grows with model size, beam count, and generated sequence length.
Diverse beam search cannot be combined with `do_sample: true`. Nucleus sampling
(`top_p`) is not implemented by the generation sampler.

With `return_dict_in_generate: true`, beam decoding returns the selected
`sequences` and `past_key_values: null`. The internal beam cache is released;
it does not correspond to the selected final sequences. A cache from an earlier
non-beam generation call can be supplied as input and is expanded across beams.

Release the model when finished:

```javascript
await model.dispose();
```
