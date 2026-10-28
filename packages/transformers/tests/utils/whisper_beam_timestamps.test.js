import { upstream1809 } from "./fixtures/whisper_timestamp_1809.js";
import { jest } from "@jest/globals";
import { PreTrainedModel } from "../../src/models/modeling_utils.js";
import { Tensor } from "../../src/utils/tensor.js";
import { WhisperForConditionalGeneration } from "../../src/models/whisper/modeling_whisper.js";
import { BeamSearchScorer } from "../../src/generation/beam_search.js";

const extract = (outputs, frames = 6) => WhisperForConditionalGeneration.prototype._extract_token_timestamps.call({ config: { median_filter_width: 1 } }, outputs, [[0, 0]], frames, 0.02, 2);
const sequences = () => new Tensor("int64", [9n, 8n, 1n, 2n, 3n, 0n, 9n, 8n, 4n, 0n, 0n, 0n], [2, 6]);

it.each([false, true])("reconstructs changing rows, query offsets, EOS and padding (uncached=%s)", (uncached) => {
  const history = [
    [0n, 1n, 0n, 1n, -1n, -1n],
    [2n, 3n, -1n, -1n, -1n, -1n],
  ];
  const steps = Array.from({ length: 4 }, (_, step) => {
    const queries = step === 0 ? 2 : uncached ? 2 + step : 1;
    const data = new Float32Array(4 * queries * 8).fill(-100);
    for (let row = 0; row < 4; ++row) {
      const offset = (row * queries + queries - 1) * 8;
      // Unselected rows and padded frames are intentionally misleading.
      data.fill(500 + row * 10 + step, offset, offset + 8);
      if (row === Number(history[0][step])) {
        data.fill(0, offset, offset + 8);
        data[offset + step * 2] = 10;
        data[offset + step * 2 + 1] = 10;
      }
    }
    if (step < 2) {
      const offset = (Number(history[1][step]) * queries + queries - 1) * 8;
      data.fill(0, offset, offset + 4);
      data[offset + step * 2] = 10;
      data[offset + step * 2 + 1] = 10;
    }
    return [new Tensor("float32", data, [4, 1, queries, 8])];
  });
  const result = extract({ sequences: sequences(), cross_attentions: steps, beam_indices: new Tensor("int64", history.flat(), [2, 6]), sequence_lengths: new Tensor("int64", [6n, 4n], [2]), num_beams: 2 }, [8, 4]);
  expect(Array.from(result.data.slice(0, 6))).toEqual([0, 0, 0, Math.fround(0.02), Math.fround(0.06), Math.fround(0.1)]);
  expect(Array.from(result.data.slice(6))).toEqual([0, 0, 0, Math.fround(0.02), 0, 0]);
  expect(Array.from(result.data).every(Number.isFinite)).toBe(true);
});

it("converts fp16 attention values numerically and handles constant variance", () => {
  const steps = Array.from({ length: 4 }, (_, i) => [new Tensor("float16", new Uint16Array(12).fill(i % 2 ? 0xbc00 : 0x3c00), [2, 1, 1, 6])]);
  const output = { sequences: sequences(), cross_attentions: steps, sequence_lengths: new Tensor("int64", [6n, 4n], [2]) };
  const half = extract(output);
  const float = extract({ ...output, cross_attentions: steps.map((step) => step.map((t) => t.to("float32"))) });
  expect(half.tolist()).toEqual(float.tolist());
  expect(Array.from(half.data).every(Number.isFinite)).toBe(true);
});

it("snapshots finished EOS ancestry using global diverse-group rows", () => {
  const scorer = new BeamSearchScorer(2, 2, { eos_token_id: 0, decoder_prompt_len: 1 });
  const history = [[4], [5], [12], [13]];
  scorer.process(
    [
      [9n, 1n],
      [9n, 2n],
      [9n, 3n],
      [9n, 4n],
    ],
    [0, 0, 0, 0],
    [0n, 1n, 2n, 0n, 1n, 2n],
    [1, 0, 1, 0, 0, 1],
    [-1, -2, -3, -1, -2, -3],
    history,
    [4, 5, 12, 13],
  );
  history[1][0] = 999;
  expect(scorer._beam_hyps[0].beams[0].beam_indices).toEqual([5, 5]);
  expect(scorer._beam_hyps[1].beams[0].beam_indices).toEqual([12, 12]);
  expect(scorer._beam_hyps[0].beams[0].tokens).toEqual([9n, 2n, 0n]);
});

it.each([
  { batch: 2, returns: 1 },
  { batch: 1, returns: 2 },
])("rejects unsupported seek combinations before forward: %j", async ({ batch, returns }) => {
  const model = Object.assign(Object.create(WhisperForConditionalGeneration.prototype), {
    _prepare_generation_config: () => ({ return_timestamps: true, return_token_timestamps: true, alignment_heads: [[0, 0]], num_return_sequences: returns }),
    _retrieve_init_tokens: () => [9, 8],
    forward: () => {
      throw Error("must not allocate");
    },
  });
  await expect(model.generate({ inputs: new Tensor("float32", new Float32Array(batch * 8), [batch, 1, 8]) })).rejects.toThrow("single input and a single returned sequence");
});

it("keeps tiny cropped recordings finite with the default median window", () => {
  const output = { sequences: new Tensor("int64", [9n, 8n, 1n, 0n], [1, 4]), cross_attentions: [[new Tensor("float32", [1], [1, 1, 1, 1])], [new Tensor("float32", [1], [1, 1, 1, 1])]] };
  const result = WhisperForConditionalGeneration.prototype._extract_token_timestamps.call({ config: { median_filter_width: 7 } }, output, [[0, 0]], 1, 0.02, 2);
  expect(result.tolist()).toEqual([[0, 0, 0, 0]]);
});

it("keeps active ancestry when finalizing at the token limit", () => {
  const scorer = new BeamSearchScorer(1, 2, { decoder_prompt_len: 2, num_return_sequences: 2 });
  const history = [
    [1, 0],
    [0, 1],
  ];
  const results = scorer.finalize_with_scores(
    [
      [9n, 8n, 2n, 3n],
      [9n, 8n, 4n, 5n],
    ],
    [-2, -1],
    history,
  );
  history[1][1] = 99;
  expect(results.map((x) => x.beam_indices)).toEqual([
    [0, 1],
    [1, 0],
  ]);
  expect(results.map((x) => x.tokens.at(-1))).toEqual([5n, 3n]);
});

it.each([false, true])("crops retained seek segments before alignment and releases scratch (failure=%s)", async (fail) => {
  const owned = [];
  const calls = [];
  let segment = 0;
  const forward = jest.spyOn(PreTrainedModel.prototype, "generate").mockImplementation(async () => {
    const ids = segment++ === 0 ? [9n, 100n, 1n, 102n, 102n, 2n, 0n] : [9n, 100n, 2n, 102n, 0n];
    const sequences = new Tensor("int64", ids, [1, ids.length]);
    const attention = new Tensor("float32", [1, 2, 3, 4], [1, 1, 1, 4]);
    owned.push(jest.spyOn(sequences, "dispose"), jest.spyOn(attention, "dispose"));
    return { sequences, cross_attentions: [[attention]] };
  });
  const model = Object.assign(Object.create(WhisperForConditionalGeneration.prototype), {
    config: { max_source_positions: 4 },
    _extract_token_timestamps(outputs, heads, frames) {
      calls.push({ frames, length: Number(outputs.sequence_lengths.data[0]) });
      if (fail) throw Error("alignment failed");
      return new Tensor("float32", new Float32Array(outputs.sequences.dims[1]), outputs.sequences.dims);
    },
  });
  try {
    const result = model._generate_with_seek({ inputs: new Tensor("float32", new Float32Array(8), [1, 1, 8]), generation_config: { no_timestamps_token_id: 99, eos_token_id: 0, return_token_timestamps: true, alignment_heads: [[0, 0]], num_beams: 2 }, init_tokens: [9], kwargs: {} });
    if (fail) await expect(result).rejects.toThrow("alignment failed");
    else {
      const output = await result;
      expect(calls).toEqual([
        { frames: 2, length: 4 },
        { frames: 2, length: 4 },
      ]);
      expect(output.sequences.tolist()).toEqual([[9n, 100n, 1n, 102n, 102n, 2n, 104n, 0n]]);
    }
    for (const dispose of owned) expect(dispose).toHaveBeenCalledTimes(1);
  } finally {
    forward.mockRestore();
  }
});

it("expands an explicitly supplied flat prefix across the short-form batch", async () => {
  const forward = jest.spyOn(PreTrainedModel.prototype, "generate").mockImplementation(async ({ decoder_input_ids }) => {
    expect(decoder_input_ids).toEqual([
      [9, 8],
      [9, 8],
    ]);
    return new Tensor("int64", [9n, 8n, 1n, 9n, 8n, 1n], [2, 3]);
  });
  const model = Object.assign(Object.create(WhisperForConditionalGeneration.prototype), {
    _prepare_generation_config: () => ({ max_new_tokens: 1 }),
  });
  try {
    await model.generate({ input_features: new Tensor("float32", new Float32Array(16), [2, 1, 8]), decoder_input_ids: [9, 8] });
    expect(forward).toHaveBeenCalledTimes(1);
  } finally {
    forward.mockRestore();
  }
});

// Compare against the independent, pre-beam PR #1809 algorithm.
const referenceFixture = (frames = 6) => {
  let seed = 1234;
  const rows = Array.from({ length: 5 }, () =>
    Float32Array.from({ length: frames }, () => {
      seed = (1664525 * seed + 1013904223) >>> 0;
      return seed / 2 ** 32;
    }),
  );
  return {
    sequences: new Tensor("int64", [9n, 8n, 1n, 2n, 3n, 0n], [1, 6]),
    cross_attentions: [[new Tensor("float32", Float32Array.from([...rows[0], ...rows[1]]), [1, 1, 2, frames])], ...rows.slice(2).map((row) => [new Tensor("float32", row, [1, 1, 1, frames])])],
  };
};
it.each([1, 3])("matches PR #1809 single-beam alignment including prefix normalization (median=%s)", (median) => {
  const output = referenceFixture(12);
  const config = { median_filter_width: median, decoder_layers: 1 };
  const actual = () => WhisperForConditionalGeneration.prototype._extract_token_timestamps.call({ config }, output, [[0, 0]], 12, 0.02, 2).tolist();
  const expected = () => upstream1809.call({ config }, output, [[0, 0]], 12, 0.02, 2).tolist();
  expect(actual()).toEqual(expected());
  output.cross_attentions[0][0].data.set([5, -8, 12, 0, 1, 20, -4, 2, 19, -5, 1, 9]);
  expect(actual()).toEqual(expected());
});
it.each([2, 3])("does not median-filter %s frames with a seven-frame window", (frames) => {
  const output = referenceFixture(frames);
  const filtered = WhisperForConditionalGeneration.prototype._extract_token_timestamps.call({ config: { median_filter_width: 7 } }, output, [[0, 0]], frames, 0.02, 2);
  expect(filtered.tolist()).toEqual(extract(output, frames).tolist());
});
it("converts only selected fp16 attention elements", () => {
  const output = referenceFixture();
  output.cross_attentions = output.cross_attentions.map((step) => step.map((t) => new Tensor("float16", new Uint16Array(4 * 3 * t.dims[2] * 6).fill(0x3c00), [4, 3, t.dims[2], 6])));
  let elements = 0;
  const original = Tensor.prototype.to;
  const spy = jest.spyOn(Tensor.prototype, "to").mockImplementation(function (type) {
    if (this.type === "float16" && type === "float32") elements += this.data.length;
    return original.call(this, type);
  });
  try {
    expect(extract(output).data.every(Number.isFinite)).toBe(true);
    expect(elements).toBe(5 * 6);
  } finally {
    spy.mockRestore();
  }
});

it("preserves preprocessed silence padding while bounding seek to valid audio frames", async () => {
  const inputs = new Tensor("float32", [1, 2, 3, -1.5, -1.5, -1.5, -1.5, -1.5], [1, 1, 8]);
  const forward = jest.spyOn(PreTrainedModel.prototype, "generate").mockImplementation(async ({ input_features }) => {
    expect(input_features.tolist()).toEqual(inputs.tolist());
    return new Tensor("int64", [9n, 100n, 2n, 101n, 0n], [1, 5]);
  });
  const model = Object.assign(Object.create(WhisperForConditionalGeneration.prototype), { config: { max_source_positions: 4 } });
  try {
    const output = await model._generate_with_seek({ inputs, generation_config: { num_frames: 3, no_timestamps_token_id: 99, eos_token_id: 0, return_token_timestamps: false }, init_tokens: [9], kwargs: {} });
    expect(forward).toHaveBeenCalledTimes(1);
    expect(output.tolist()).toEqual([[9n, 100n, 2n, 101n, 0n]]);
    expect(inputs.tolist()[0][0][7]).toBe(-1.5);
  } finally {
    forward.mockRestore();
  }
});

// Reconstruct each hypothesis independently and compare with the upstream
// non-beam oracle, including early EOS and a second selected attention head.
it.each([1, 7])("matches PR #1809 for selected beam histories (median=%s)", (median) => {
  const histories = [
    [0, 1, 0, 1, 1],
    [2, 3, 2],
  ];
  const steps = Array.from({ length: 5 }, (_, step) =>
    Array.from({ length: 2 }, (_, layer) => {
      const dims = [4, 2, step === 0 ? 3 : 1, 16];
      return new Tensor(
        "float32",
        Float32Array.from({ length: dims.reduce((a, b) => a * b) }, (_, i) => Math.sin(i * 0.83 + step * 2.31 + layer) * 0.6 + Math.cos(i * 0.12 + step) * 0.3),
        dims,
      );
    }),
  );
  const outputs = {
    sequences: new Tensor(
      "int64",
      Array.from({ length: 16 }, (_, i) => BigInt(i)),
      [2, 8],
    ),
    cross_attentions: steps,
    beam_indices: new Tensor("int64", histories.flatMap((row) => [...row, ...Array(8 - row.length).fill(-1)]).map(BigInt), [2, 8]),
    sequence_lengths: new Tensor("int64", [8n, 6n], [2]),
    num_beams: 2,
  };
  const config = { median_filter_width: median, decoder_layers: 2 };
  const heads = [
    [0, 0],
    [1, 1],
  ];
  const actual = WhisperForConditionalGeneration.prototype._extract_token_timestamps.call({ config }, outputs, heads, [16, 10], 0.02, 3).tolist();
  for (let result = 0; result < 2; ++result) {
    const history = histories[result];
    const single = { sequences: new Tensor("int64", Array(history.length + 3).fill(1n), [1, history.length + 3]), cross_attentions: history.map((row, step) => steps[step].map((t) => t.slice([row, row + 1]))) };
    const expected = upstream1809.call({ config }, single, heads, [16, 10][result], 0.02, 3).tolist()[0];
    expect(actual[result].slice(0, expected.length)).toEqual(expected);
    expect(actual[result].slice(expected.length).every((x) => x === 0)).toBe(true);
  }
});

it.each([0n, 7n])("aligns the final predicting query at EOS or token limit (%s)", (lastToken) => {
  const output = referenceFixture(12);
  output.sequences.data[5] = lastToken;
  const config = { median_filter_width: 1, decoder_layers: 1 };
  expect(extract(output, 12).tolist()).toEqual(upstream1809.call({ config }, output, [[0, 0]], 12, 0.02, 2).tolist());
});
it("handles EOS on the first forward without a subsequent query", () => {
  const output = { sequences: new Tensor("int64", [9n, 8n, 0n], [1, 3]), cross_attentions: referenceFixture().cross_attentions.slice(0, 1), beam_indices: new Tensor("int64", [0n, -1n, -1n], [1, 3]), sequence_lengths: new Tensor("int64", [3n], [1]), num_beams: 1 };
  expect(extract(output).tolist()).toEqual([[0, 0, 0]]);
});
