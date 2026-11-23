import { existsSync } from "node:fs";
import path from "node:path";
import { env } from "../src/env.js";
import { makeSamples, collate, revisions } from "./helpers/multimodal.js";
import { jest } from "@jest/globals";
import { LlavaForConditionalGeneration, LlavaProcessor, Idefics3ForConditionalGeneration, Idefics3Processor, Qwen2VLForConditionalGeneration, Qwen2VLProcessor, RawImage, Tensor, cat, random } from "../src/transformers.js";
import { init } from "./init.js";
init();
const models = [
  ["llava", LlavaForConditionalGeneration, LlavaProcessor, "Xenova/tiny-random-LlavaForConditionalGeneration"],
  ["idefics3", Idefics3ForConditionalGeneration, Idefics3Processor, "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration"],
  ["qwen2_vl", Qwen2VLForConditionalGeneration, Qwen2VLProcessor, "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"],
];
const image = (value) => new RawImage(new Uint8ClampedArray(56 * 56 * 3).fill(value), 56, 56, 3);
// Collate independently processed examples to test actual left padding, independent
// of processor batching support. Image features remain in original sample order.
function batchInputs(samples, family) {
  const length = Math.max(...samples.map((x) => x.input_ids.dims[1]));
  const output = {};
  for (const key of ["input_ids", "attention_mask"]) {
    output[key] = new Tensor(
      "int64",
      samples.flatMap((x) => [...new Array(length - x[key].dims[1]).fill(0n), ...x[key].data]),
      [samples.length, length],
    );
  }
  for (const key of ["pixel_values", "pixel_attention_mask", "image_grid_thw"]) {
    if (samples[0][key])
      output[key] = cat(
        samples.map((x) => x[key]),
        0,
      );
  }
  return output;
}
for (const [family, Model, Processor, model_id] of models) {
  describe(`Real multimodal generation: ${family}`, () => {
    let model, processor, samples, batch;
    beforeAll(async () => {
      const cached = path.join(env.cacheDir, model_id);
      const source = existsSync(path.join(cached, "config.json")) ? cached : model_id;
      const options = { revision: revisions[family], dtype: "fp32", session_options: { intraOpNumThreads: 1 } };
      model = await Model.from_pretrained(source, options);
      processor = await Processor.from_pretrained(source, options);
      samples = [];
      for (let i = 0; i < 2; ++i) {
        const text = family === "llava" ? `<image>${i ? "Describe this image in detail." : "Hi"}` : processor.apply_chat_template([{ role: "user", content: [{ type: "image" }, { type: "text", text: i ? "Describe this image in detail." : "Hi" }] }], { add_generation_prompt: true });
        samples.push(family === "llava" ? await processor(image(i * 255), text) : await processor(text, image(i * 255), { do_image_splitting: false }));
      }
      batch = batchInputs(samples, family);
    }, 120000);
    it("matches cached continuation and caller-supplied embeddings", async () => {
      const sample = samples[0];
      const options = { do_sample: false, eos_token_id: null, max_new_tokens: 2 };
      const prefill = await model.generate({ ...sample, ...options, return_dict_in_generate: true });
      const attention_mask = cat([sample.attention_mask, new Tensor("int64", [1n, 1n], [1, 2])], 1);
      const continuation = { ...sample, input_ids: prefill.sequences, attention_mask, ...options, num_beams: 2 };
      const continuationEncoder = jest.spyOn(model, "encode_image");
      let cached;
      try {
        cached = await model.generate({ ...continuation, past_key_values: prefill.past_key_values });
        expect(continuationEncoder).not.toHaveBeenCalled();
      } finally {
        continuationEncoder.mockRestore();
      }
      const uncached = await model.generate(continuation);
      expect(cached.tolist()).toEqual(uncached.tolist());
      await prefill.past_key_values.dispose();

      const inputs_embeds = await model.encode_text(sample);
      const image_features = await model.encode_image(sample);
      const merged = model._merge_input_ids_with_image_features({ ...sample, inputs_embeds, image_features });
      image_features.dispose();
      const expected = await model.generate({ ...sample, ...options, num_beams: 2 });
      const encoder = jest.spyOn(model, "encode_image");
      try {
        const actual = await model.generate({ ...sample, ...merged, ...options, num_beams: 2 });
        expect(actual.tolist()).toEqual(expected.tolist());
        expect(encoder).not.toHaveBeenCalled();
        expect(inputs_embeds.data.length).toBeGreaterThan(0);
      } finally {
        encoder.mockRestore();
        inputs_embeds.dispose();
      }
    }, 120000);
    it(`merges a new image prompt following a supplied text-prefix cache (${family === "idefics3" ? "greedy: export GQA limits cached multi-token batches" : "two beams"})`, async () => {
      const prefix = await model.generate({ input_ids: new Tensor("int64", [1n, 2n], [1, 2]), attention_mask: new Tensor("int64", [1n, 1n], [1, 2]), max_new_tokens: 1, do_sample: false, eos_token_id: null, return_dict_in_generate: true });
      try {
        const input_ids = cat([prefix.sequences, samples[0].input_ids], 1);
        const attention_mask = cat([new Tensor("int64", [1n, 1n, 1n], [1, 3]), samples[0].attention_mask], 1);
        // This tiny Idefics3 export requires batch_size=1 for multi-token cached prefill.
        const options = { ...samples[0], input_ids, attention_mask, num_beams: family === "idefics3" ? 1 : 2, max_new_tokens: 2, do_sample: false, eos_token_id: null };
        const encoder = jest.spyOn(model, "encode_image");
        let cached;
        try {
          cached = await model.generate({ ...options, past_key_values: prefix.past_key_values });
          expect(encoder).toHaveBeenCalledTimes(1);
          expect(encoder.mock.calls[0][0].pixel_values).toBe(samples[0].pixel_values);
          // Tiny random models can emit identical tokens even when vision is
          // silently skipped. Inspect the actual merged decoder embeddings.
          const embeddings = [];
          const originalForward = model.forward.bind(model);
          const forward = jest.spyOn(model, "forward").mockImplementation(async function (inputs) {
            if (inputs.inputs_embeds) embeddings.push(Array.from(inputs.inputs_embeds.data));
            return originalForward(inputs);
          });
          try {
            encoder.mockClear();
            for (const pixel_values of [samples[0].pixel_values, samples[1].pixel_values]) {
              await model.generate({ ...options, pixel_values, past_key_values: prefix.past_key_values });
            }
            expect(encoder).toHaveBeenCalledTimes(2);
            expect(embeddings).toHaveLength(2);
            expect(embeddings[0]).not.toEqual(embeddings[1]);
          } finally {
            forward.mockRestore();
          }
        } finally {
          encoder.mockRestore();
        }
        const reference = await model.generate(options);
        expect(cached.tolist()).toEqual(reference.tolist());
      } finally {
        await prefix.past_key_values.dispose();
      }
    }, 120000);
    if (family === "qwen2_vl")
      it("rejects raw new images after an image-containing cache before encoding", async () => {
        const prefix = await model.generate({ ...samples[0], max_new_tokens: 1, eos_token_id: null, return_dict_in_generate: true });
        const encode = jest.spyOn(model, "encode_image");
        try {
          const input_ids = cat([prefix.sequences, samples[1].input_ids], 1);
          const attention_mask = new Tensor("int64", new BigInt64Array(input_ids.size).fill(1n), input_ids.dims);
          // Full-history grids are required for rotary positions, but raw features
          // cannot safely be selected from old and new image groups yet.
          await expect(
            model.generate({
              ...samples[1],
              input_ids,
              attention_mask,
              image_grid_thw: cat(
                samples.map((sample) => sample.image_grid_thw),
                0,
              ),
              past_key_values: prefix.past_key_values,
              num_beams: 2,
              max_new_tokens: 2,
            }),
          ).rejects.toThrow("cannot append raw images after an image-containing cache");
          await expect(model.generate({ ...samples[0], past_key_values: prefix.past_key_values, num_beams: 2, max_new_tokens: 2 })).rejects.toThrow("cached raw-image prompts require full input_ids history");
          expect(encode).not.toHaveBeenCalled();
        } finally {
          encode.mockRestore();
          await prefix.past_key_values.dispose();
        }
      }, 120000);
    it("preserves unequal image groups with two beams", async () => {
      const ragged = await makeSamples(processor, family);
      const batch = collate(ragged);
      const options = { num_beams: 2, max_new_tokens: 2, do_sample: false, repetition_penalty: 2, eos_token_id: null };
      const result = await model.generate({ ...batch, ...options });
      const expected = [];
      for (let i = 0; i < ragged.length; ++i) {
        const single = await model.generate({ ...ragged[i], input_ids: batch.input_ids.slice([i, i + 1]), attention_mask: batch.attention_mask.slice([i, i + 1]), ...options });
        expected.push(...single.tolist().map((row) => row.slice(batch.input_ids.dims[1])));
      }
      expect(result.tolist().map((row) => row.slice(batch.input_ids.dims[1]))).toEqual(expected);
    }, 120000);
    afterAll(async () => {
      await model?.dispose();
    }, 10000);
    it.each([1, 2, 4])(
      "matches independent outputs with padding and %s beams",
      async (num_beams) => {
        const options = { do_sample: false, repetition_penalty: 2, num_beams, max_new_tokens: 3, eos_token_id: null, num_return_sequences: num_beams > 1 ? 2 : 1 };
        const encoder = jest.spyOn(model, "encode_image");
        const result = await model.generate({ ...batch, ...options });
        expect(encoder).toHaveBeenCalledTimes(1);
        encoder.mockRestore();
        const expected = [];
        for (let i = 0; i < samples.length; ++i) {
          const sample = { ...samples[i], input_ids: batch.input_ids.slice([i, i + 1]), attention_mask: batch.attention_mask.slice([i, i + 1]) };
          const single = await model.generate({ ...sample, ...options });
          expected.push(...single.tolist().map((row) => row.slice(sample.input_ids.dims[1])));
        }
        expect(result.tolist().map((row) => row.slice(batch.input_ids.dims[1]))).toEqual(expected);
      },
      120000,
    );
    it("runs diverse search and reproducible seeded beam sampling", async () => {
      const diverse = await model.generate({ ...samples[0], do_sample: false, max_new_tokens: 2, num_beams: 4, num_beam_groups: 2, diversity_penalty: 1, num_return_sequences: 2 });
      expect(diverse.dims).toEqual([2, samples[0].input_ids.dims[1] + 2]);
      const options = { ...batch, max_new_tokens: 2, num_beams: 2, do_sample: true, top_k: 10 };
      random.seed(123);
      const a = await model.generate(options);
      random.seed(123);
      const b = await model.generate(options);
      expect(a.tolist()).toEqual(b.tolist());
    }, 120000);
  });
}
