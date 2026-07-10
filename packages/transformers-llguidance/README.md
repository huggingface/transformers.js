# @huggingface/transformers-llguidance

Experimental constrained-generation helpers for Transformers.js.

This package exports `LlguidanceConstraint`, which turns an llguidance response format into the `logits_processor` and `stopping_criteria` objects accepted by Transformers.js generation.

```js
import { LlguidanceConstraint } from "@huggingface/transformers-llguidance";

const { logits_processor, stopping_criteria } =
  await LlguidanceConstraint.fromResponseFormat(tokenizer, {
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

await model.generate({
  ...inputs,
  logits_processor,
  stopping_criteria,
});
```
