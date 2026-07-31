# @huggingface/transformers-llguidance

Experimental constrained-generation helpers for Transformers.js.

This package exports `LlguidanceConstraint`, which turns an llguidance response format into the `logits_processor` and `stopping_criteria` objects accepted by Transformers.js generation.

```js
import { LlguidanceConstraint } from "@huggingface/transformers-llguidance";

const constraint = await LlguidanceConstraint.fromResponseFormat(tokenizer, {
  type: "json_schema",
  json_schema: {
    type: "object",
    properties: {
      answer: { type: "string" },
    },
    required: ["answer"],
    additionalProperties: false,
  },
});

try {
  await model.generate({
    ...inputs,
    logits_processor: constraint.logits_processor,
    stopping_criteria: constraint.stopping_criteria,
  });
} finally {
  constraint.dispose();
}
```

Constraints currently support a single generated sequence at a time. Generation throws when the logical batch size is not `1` rather than sharing mutable grammar state across sequences.

The interpreter is automatically disposed when llguidance reaches a terminal state. Call `dispose()` in a `finally` block as shown above to also release resources when generation ends for another reason, such as `max_new_tokens` or cancellation.

If llguidance reports acceptance before sampling, the constraint forces the tokenizer's EOS token so no unconstrained token is appended. Compatible tokenizer objects must expose an EOS ID such as `eos_token_id` or `eosTokenId`; generation fails closed if acceptance occurs without one.

## Regex constraints

Use `type: "regex"` to constrain generation to a regular expression. For example, this only allows ISO-like dates in `YYYY-MM-DD` format:

```js
import { LlguidanceConstraint } from "@huggingface/transformers-llguidance";

const constraint = await LlguidanceConstraint.fromResponseFormat(tokenizer, {
  type: "regex",
  regex: "\\d{4}-\\d{2}-\\d{2}",
});

try {
  const output = await model.generate({
    ...inputs,
    logits_processor: constraint.logits_processor,
    stopping_criteria: constraint.stopping_criteria,
  });
} finally {
  constraint.dispose();
}
```
