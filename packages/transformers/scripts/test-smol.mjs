import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
} from "../src/transformers.js";

const MODEL_ID = process.env.SMOLVLM_MODEL_ID ?? "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";
const MAX_NEW_TOKENS = Number(process.env.SMOLVLM_MAX_NEW_TOKENS ?? "16");
const messages = [{
  role: "user",
  content: [
    { type: "image" },
    { type: "text", text: "Can you describe this image?" },
  ],
}];
const IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg";

let modelId = MODEL_ID;
console.log(`Loading processor and model from: ${modelId}`);
const processor = await AutoProcessor.from_pretrained(modelId);
const model = await AutoModelForImageTextToText.from_pretrained(modelId, { dtype: "fp32" });
const text = processor.apply_chat_template(messages, {
  add_generation_prompt: true,
});
const image = await RawImage.fromURL(IMAGE_URL);
const inputs = await processor(text, image);
const generated_ids = await model.generate({
  ...inputs,
  do_sample: false,
  max_new_tokens: MAX_NEW_TOKENS,
});
const generated_text = processor.batch_decode(generated_ids, { skip_special_tokens: true })[0];
console.log("Model class:", model.constructor.name);
console.log("Generated text preview:\n", generated_text);
await model.dispose();
