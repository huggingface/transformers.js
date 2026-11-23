# @huggingface/transformers-structured-output

Experimental constrained-generation helpers for Transformers.js.

This dependency-free package exports `StructuredOutputProcessor`, which constrains Transformers.js generation to a JSON schema, JSON object, or regular expression.

The constraint engine is implemented specifically for Transformers.js and has no runtime dependencies.

```js
import { StructuredOutputProcessor } from "@huggingface/transformers-structured-output";

const processor = new StructuredOutputProcessor(tokenizer, {
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
  logits_processor: processor,
});
```

Constraints currently support a single generated sequence at a time. Generation throws when the logical batch size is not `1` rather than sharing mutable grammar state across sequences.

A processor tracks the state of one generation. Create a new `StructuredOutputProcessor` for every `generate()` or pipeline call; constructing one is cheap once the tokenizer has been prepared (see `StructuredOutputProcessor.warmup()`).

### Thinking models

When the tokenizer defines `response_template.fields.reasoning_content` and `response_template.fields.content`, the processor understands the template's thinking region. Inside thinking, the body is unconstrained, and the requested constraint applies when final content begins. The prompt is replayed from the last `start_anchor` (or `start_anchor_pattern`) match, so prefilled thinking blocks and content openers are taken into account. Up to four whitespace characters are allowed between a closed thinking block (or content opener) and the constrained answer, matching templates that emit `</think>\n\n` before the answer.

The `enable_thinking` option controls whether the model may think before the answer:

- omitted (default): thinking only continues if the prompt already started it, for example a prefilled, unclosed `<think>` or a partially written thinking opener. Otherwise the constraint applies from the first generated token and the thinking opener is not allowed.
- `false`: the constraint applies from the first generated token. A prompt that prefills an unfinished thinking region raises an error.
- `true`: the model may open the thinking region before the constrained answer.

Thinking is opt-in because some templates (for example, Gemma 4) write no marker when thinking is off, so the prompt alone cannot tell whether thinking is disabled. Completed thinking blocks in the prompt (Qwen3's `<think>\n\n</think>\n\n` with thinking off) are accepted with every value, and invalid answer prefills raise errors. Once content satisfies the constraint, the processor allows the response template's content closer or the model's EOS token.

For models that should reason before answering, pass `{ enable_thinking: true }` to the processor and use the same flag when rendering the prompt:

```js
const enable_thinking = true;
const inputs = tokenizer.apply_chat_template(messages, {
  add_generation_prompt: true,
  return_dict: true,
  enable_thinking,
});
const processor = new StructuredOutputProcessor(tokenizer, responseFormat, { enable_thinking });
await model.generate({ ...inputs, logits_processor: processor });
```

Thinking close delimiters must be literals (or a list of literals). Openers, content closers and start anchors may be literals or patterns supported by the constraint regex engine, except that protocol patterns containing positional `^`/`$` assertions, and start-anchor patterns containing ordered `|` alternatives outside character classes, fall back with a warning. The protocol's language matcher cannot reproduce the response parser's positional and ordered-anchor-search semantics for those patterns. Escaped literal assertions/alternation characters remain supported; delimiter alternatives retain the existing longest-accepted-endpoint behavior. This restriction does not change ordinary payload regex constraints. Unsupported templates fall back to constraining the response from its first token, without interpreting thinking delimiters or prompt anchors.

Start-anchor patterns that accept the empty string (for example `a*` or `a?`) also warn and fall back:
their empty-match selection is not represented by the protocol search. Nullable payload regexes remain
supported. Special content-closer tokens select the delimiter branch on commit, even when the same
bytes could extend the payload regex; ordinary tokens with those bytes keep their payload meaning.

## Supported constraints

The JSON engine implements a practical JSON Schema 2020-12 profile. This includes deep `const` and `enum`, exact decimal bounds and `multipleOf`, tuple and homogeneous arrays, `contains`, deep `uniqueItems`, object property and dependency assertions, `allOf`/`anyOf`/`oneOf`/`not`, conditionals, local `$ref`, recursive `$defs`, draft-07 compatibility, and root-level `x-guidance` separators. String `pattern` and recognized `format` assertions are rejected because they cannot be enforced incrementally; unknown format names remain annotations. External and dynamic references and unevaluated-property/item assertions also remain unsupported.

The regex engine performs full-string matching and supports literals, UTF-8 literals, alternation, groups, character classes, `.`, `\\d`, `\\s`, `\\w`, `^` and `$` anchors at the start and end of the pattern or of a top-level alternative (`^yes$|^no$`), and greedy `*`, `+`, `?`, and `{m,n}` quantifiers. Lookarounds, backreferences, lazy quantifiers, and Unicode character classes are intentionally unsupported.

When the generated bytes satisfy the constraint, the processor allows the tokenizer's EOS token. Transformers.js then stops generation through its built-in EOS stopping criterion, so the model generation config and tokenizer must use the same `eos_token_id`.

## Regex constraints

Use `type: "regex"` to constrain generation to a regular expression. For example, this only allows ISO-like dates in `YYYY-MM-DD` format:

```js
import { StructuredOutputProcessor } from "@huggingface/transformers-structured-output";

const processor = new StructuredOutputProcessor(tokenizer, {
  type: "regex",
  regex: "\\d{4}-\\d{2}-\\d{2}",
});

const output = await model.generate({
  ...inputs,
  logits_processor: processor,
});
```
