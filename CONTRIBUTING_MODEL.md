# Adding a New Model to Transformers.js

This guide walks through everything needed to add support for a new model architecture. The process has two main phases: **exporting the model to ONNX**, then **wiring it into the library**.

## Table of Contents

1. [Export the Model to ONNX](#1-export-the-model-to-onnx)
2. [Add the Model to the Library](#2-add-the-model-to-the-library)
3. [Write Tests](#3-write-tests)

---

## 1. Export the Model to ONNX

Transformers.js runs models using ONNX Runtime. Before adding a model to the library, you need an ONNX export of it.

- For LLMs, we recommend exporting with [https://github.com/microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)
- For other models, we recommend exporting with [https://github.com/huggingface/optimum-onnx](https://github.com/huggingface/optimum-onnx)

Once exported, upload the ONNX files to the Hugging Face Hub alongside the model's original config and tokenizer files so they can be loaded with `from_pretrained`.

---

## 2. Add the Model to the Library

Every model in Transformers.js is built from the same pieces:

- **A model class**: extends `PreTrainedModel`, which handles all ONNX inference, generation, and KV-cache management
- **Task head classes**: thin wrappers that wrap the output in the right output object (e.g. `MaskedLMOutput`)
- **A tokenizer and/or processor**: only needed if the model requires a custom one; most models reuse an existing class

All model files live under `packages/transformers/src/models/<model_type>/`. Look at an existing model of the same type to understand what's needed, Most are just a few lines.

### Model class

Every model file exports a base class and one or more task heads. For the vast majority of models, these are empty subclasses. All the logic lives in `PreTrainedModel`.

**Decoder-only LLM:**

```js
import { PreTrainedModel } from '../modeling_utils.js';

export class MyModelPreTrainedModel extends PreTrainedModel {}
export class MyModelModel extends MyModelPreTrainedModel {}
export class MyModelForCausalLM extends MyModelPreTrainedModel {}
```

**Encoder-only model:**

```js
import { PreTrainedModel } from '../modeling_utils.js';
import { MaskedLMOutput, SequenceClassifierOutput } from '../modeling_outputs.js';

export class MyModelPreTrainedModel extends PreTrainedModel {}
export class MyModelModel extends MyModelPreTrainedModel {}

export class MyModelForMaskedLM extends MyModelPreTrainedModel {
    async _call(model_inputs) {
        return new MaskedLMOutput(await super._call(model_inputs));
    }
}

export class MyModelForSequenceClassification extends MyModelPreTrainedModel {
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
```

Only add the task heads the model actually supports. The available output classes (`MaskedLMOutput`, `TokenClassifierOutput`, `Seq2SeqLMOutput`, etc.) are all in `modeling_outputs.js`.

### Tokenizer and processor

Most models reuse an existing tokenizer (e.g. all Llama-family models use `LlamaTokenizer`). Only create a new one if the model genuinely needs custom tokenization or preprocessing logic.

| What | File | Barrel to update |
| --- | --- | --- |
| Custom tokenizer | `src/models/<name>/tokenization_<name>.js` | `src/models/tokenizers.js` |
| Custom image processor | `src/models/<name>/image_processing_<name>.js` | `src/models/image_processors.js` |
| Custom multimodal processor | `src/models/<name>/processing_<name>.js` | `src/models/processors.js` |
| Custom audio/feature extractor | `src/models/<name>/feature_extraction_<name>.js` | `src/models/feature_extractors.js` |

The class name must match the `tokenizer_class` or `processor_class` field in the model's `tokenizer_config.json` / `preprocessor_config.json` on the Hub.

### Wiring it up

Once the model file is written, three more files need updating:

1. **`src/models/models.js`**: add `export * from './<name>/modeling_<name>.js'`
2. **`src/models/registry.js`**: map the `model_type` string (from `config.json`) to the class names, and set the correct loading category (`EncoderOnly`, `DecoderOnly`, `Seq2Seq`, etc.)
3. **`src/configs.js`**: for generative models, add a `case` in `getNormalizedConfig()` to map the model's config field names to the normalized names the KV-cache runtime expects

Look at a similar existing model in each file to see exactly what to add.

---

## 3. Write Tests

Create `packages/transformers/tests/models/<model_type>/test_modeling_<model_type>.js`. The test runner auto-discovers files by this naming convention. No registration needed.

Use a small, fast model. The convention is to use a `tiny-random-*` model from `hf-internal-testing/` on the Hub. If one doesn't exist for your architecture, generate one with the `transformers` Python library:

```python
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.for_model("my_model", num_hidden_layers=2, hidden_size=64, ...)
model = AutoModelForCausalLM.from_config(config)
model.push_to_hub("hf-internal-testing/tiny-random-MyModelForCausalLM")
```

**Test file structure:**

```js
import { MyModelForCausalLM, MyModelTokenizer } from "../../../src/transformers.js";
import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
    describe("MyModelForCausalLM", () => {
        const model_id = "hf-internal-testing/tiny-random-MyModelForCausalLM";
        let model, tokenizer;

        beforeAll(async () => {
            model = await MyModelForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
            tokenizer = await MyModelTokenizer.from_pretrained(model_id);
        }, MAX_MODEL_LOAD_TIME);

        it("batch_size=1", async () => {
            const inputs = tokenizer("hello");
            const outputs = await model.generate({ ...inputs, max_length: 10 });
            expect(outputs.tolist()).toEqual([[/* expected token ids */]]);
        }, MAX_TEST_EXECUTION_TIME);

        it("batch_size>1", async () => {
            const inputs = tokenizer(["hello", "hello world"], { padding: true });
            const outputs = await model.generate({ ...inputs, max_length: 10 });
            expect(outputs.tolist()).toEqual([[...], [...]]);
        }, MAX_TEST_EXECUTION_TIME);

        afterAll(async () => { await model?.dispose(); }, MAX_MODEL_DISPOSE_TIME);
    });
};
```

Run your tests with:

```bash
# All tests
pnpm test

# Only your model's tests
pnpm --filter @huggingface/transformers test -- --testPathPattern=my_model
```
