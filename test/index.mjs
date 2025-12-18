import { pipeline, TextStreamer } from "../src/transformers.js";

// Create a text generation pipeline
const generator = await pipeline(
  "text-generation",
  "onnx-community/gemma-3-270m-it-ONNX",
  {
    dtype: "fp16",
    //progress_callback: console.log,
  },
);

// Define the list of messages
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Write a poem about machine learning." },
];

// Generate a response
const output = await generator(messages, {
  max_new_tokens: 512,
  do_sample: false,
  streamer: new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    //callback_function: console.log,
  }),
});
console.log(output[0].generated_text.at(-1).content);
