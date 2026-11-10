import { jest } from "@jest/globals";
import { RawImage, Tensor, cat } from "../../src/transformers.js";

export const revisions = {
  llava: "a61a79a541cec4898722bdfb5b016984c29fb061",
  idefics3: "5c0505e5c04d9e87b7cdb599076e6cf0cedf707d",
  qwen2_vl: "2e2ed20b74bd5157aafd082b00e87fe25b2231a6",
};

export function collate(samples) {
  const length = Math.max(...samples.map((x) => x.input_ids.dims[1]));
  const result = {};
  for (const key of ["input_ids", "attention_mask"]) {
    result[key] = new Tensor(
      "int64",
      samples.flatMap((x) => [...new Array(length - x[key].dims[1]).fill(0n), ...x[key].data]),
      [samples.length, length],
    );
  }
  for (const key of ["pixel_values", "pixel_attention_mask", "image_grid_thw", "audio_values"]) {
    if (!samples[0][key]) continue;
    let tensors = samples.map((x) => x[key]);
    if (key === "pixel_attention_mask" || (key === "pixel_values" && tensors[0].dims.length === 5)) {
      const n = Math.max(...tensors.map((t) => t.dims[1]));
      tensors = tensors.map((t) => {
        if (t.dims[1] === n) return t;
        const dims = [...t.dims];
        dims[1] = n - t.dims[1];
        return cat([t, new Tensor(t.type, new t.data.constructor(dims.reduce((a, b) => a * b, 1)), dims)], 1);
      });
    }
    result[key] = cat(tensors, 0);
  }
  return result;
}

// Exercise generation with valid preprocessed image groups independently of
// the Qwen multi-image processor fix tracked in upstream PR #1812.
async function processImages(processor, family, text, images) {
  if (family === "llava") return processor(images, text);
  if (family !== "qwen2_vl") return processor(text, images, { do_image_splitting: false });
  const imageProcessor = processor.image_processor;
  const processed = await Promise.all(images.map((image) => imageProcessor(image)));
  const imageInputs = {
    pixel_values: cat(
      processed.map((x) => x.pixel_values),
      0,
    ),
    image_grid_thw: cat(
      processed.map((x) => x.image_grid_thw),
      0,
    ),
  };
  const imageInputsStub = Object.assign(async () => imageInputs, { config: imageProcessor.config });
  const imageProcessorGetter = jest.spyOn(processor, "image_processor", "get").mockReturnValue(imageInputsStub);
  try {
    // Retain the real processor's token expansion using the supplied grids.
    return await processor(text, images);
  } finally {
    imageProcessorGetter.mockRestore();
  }
}

export async function makeSamples(processor, family) {
  const image = (value) => new RawImage(new Uint8ClampedArray(56 * 56 * 3).fill(value), 56, 56, 3);
  const samples = [];
  for (let i = 0; i < 2; ++i) {
    const images = i ? [image(255), image(127)] : [image(0)];
    const text = family === "llava" ? `${"<image>".repeat(images.length)}${i ? "Describe this image in detail." : "Hi"}` : processor.apply_chat_template([{ role: "user", content: [...images.map(() => ({ type: "image" })), { type: "text", text: i ? "Describe this image in detail." : "Hi" }] }], { add_generation_prompt: true });
    samples.push(await processImages(processor, family, text, images));
  }
  return samples;
}
