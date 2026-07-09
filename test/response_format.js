import { pipeline } from "../packages/transformers/src/transformers.js";

const colorMessages = [
  {
    role: "user",
    content:
      "Return a JSON array of three short color names. Do not include any extra text.",
  },
];

const colorSchema = {
  type: "array",
  items: { type: "string" },
};

const bookMessages = [
  {
    role: "user",
    content:
      "Return a JSON array of three classic science fiction books. Include the title, author, and a short one-sentence summary for each book. Do not include any extra text.",
  },
];

const bookSchema = {
  type: "array",
  items: {
    type: "object",
    properties: {
      title: { type: "string" },
      author: { type: "string" },
      summary: { type: "string" },
    },
    required: ["title", "author", "summary"],
    additionalProperties: false,
  },
};

let progress = 0;
const pipe = await pipeline(
  "text-generation",
  "onnx-community/gemma-4-E2B-it-ONNX",
  {
    device: "webgpu",
    dtype: "q4f16",
    progress_callback: (i) => {
      if (i.status === "progress_total") {
        const p = Math.round(i.progress);
        if (p !== progress) {
          console.log(p);
          progress = p;
        }
      }
    },
  },
);

try {
  for (const { label, messages, schema, max_new_tokens } of [
    {
      label: "Colors",
      messages: colorMessages,
      schema: colorSchema,
      max_new_tokens: 1024,
    },
    {
      label: "Books",
      messages: bookMessages,
      schema: bookSchema,
      max_new_tokens: 2048,
    },
  ]) {
    const output = await pipe(messages, {
      max_new_tokens,
      response_format: { type: "json_schema", json_schema: schema },
    });

    const generated = output[0].generated_text.at(-1).content;
    console.log(`\n${label}:`);
    console.log(generated);
    console.log(JSON.parse(generated));
  }
} finally {
  await pipe.dispose();
}
