# Typed decisions (experimental)

> [!WARNING]
> The `decision` pipeline is a port of the experimental `decision` pipeline of [🤗 Transformers](https://github.com/huggingface/transformers/pull/50054): its API may change at any time.

The `decision` pipeline answers typed questions about an input (the `state`) with a probability for every option, from one forward pass per prompt. Nothing is generated: the score of each option is the next-token logit of its label (`A`, `B`, ... or `Yes`/`No`), so any chat model can answer. Requests and answers use the System One format.

## Basic usage

```js
import { pipeline } from "@huggingface/transformers";

const decide = await pipeline("decision", "onnx-community/Qwen2.5-0.5B-Instruct", { dtype: "q4" });

const answers = await decide({
  state: "I was charged twice for my order and want a refund.",
  questions: {
    intent: {
      type: "choice",
      instructions: "What does the customer want?",
      criteria: { refund: "wants money back", cancel: "wants to cancel an order", track: "wants to track an order" },
    },
    refund: { type: "noul", instructions: "Is a refund requested?" },
    anger: { type: "score", instructions: "How angry?", criteria: ["calm", "annoyed", "furious"] },
  },
});

answers.intent; // { type: "choice", choice: "refund", confidence, probabilities: { refund, cancel, track } }
answers.refund; // { type: "noul", noul } (the probability of true)
answers.anger;  // { type: "score", score, legend: { 0: "calm", 1: "annoyed", 2: "furious" }, confidence, probabilities }
```

| Type | `criteria` | Answer |
| --- | --- | --- |
| `choice` | An object mapping each option to its description (or `null`) | `choice`, `confidence`, `probabilities` |
| `score` | The levels, lowest first | `score` (the expected level), `legend`, `confidence`, `probabilities` |
| `noul` | Optional descriptions of `true` and `false` | `noul` (the probability of `true`) |

## Decision config

A model repo can ship a `decision.json` (labels, temperatures) and a `decision_template.jinja`, which the pipeline loads with the model. You can also pass your own:

```js
import { DecisionConfig, DecisionPipeline } from "@huggingface/transformers";

const decision_config = new DecisionConfig({
  temperature: { choice: 0.7 },
  chat_template: "{{ state }}\n{{ question.instructions }}\nAnswer:<|mark|>",
});
const custom = new DecisionPipeline({ task: "decision", model: decide.model, tokenizer: decide.tokenizer, decision_config });
```

In a decision template, each answer is read at the token before a `<|mark|>`, and `<|split|>` splits the encoding. A template that reads `questions` instead of `question` renders one prompt for all the questions.

Not supported yet: decision heads (`XxxForDecision`), LoRA adapters and image `assets`.
