import { NemoConformerForTDT, Tensor } from "../../../src/transformers.js";
import { createAudioCacheKey, FeatureLRUCache } from "../../../src/models/nemo_conformer_tdt/transducer_cache.js";
import { computeTemporalDeltas } from "../../../src/models/nemo_conformer_tdt/transducer_deltas.js";

import { MAX_TEST_EXECUTION_TIME } from "../../init.js";

class MockNemoConformerForTDT extends NemoConformerForTDT {
  constructor(config, sessions, decoderScript) {
    super(config, sessions, {});
    this.decoderScript = decoderScript;
    this.decoderCalls = 0;
  }

  async _runEncoder() {
    return {
      outputs: new Tensor(
        "float32",
        new Float32Array([
          // D=2, T=3 (BDT)
          0.1,
          0.2,
          0.3, // d0 over t
          0.4,
          0.5,
          0.6, // d1 over t
        ]),
        [1, 2, 3],
      ),
    };
  }

  async _runDecoder() {
    const step = this.decoderScript[this.decoderCalls++];
    const stateShape = [1, 1, 2];
    return {
      outputs: new Tensor("float32", new Float32Array(step.logits), [1, 1, step.logits.length]),
      output_states_1: new Tensor("float32", new Float32Array([this.decoderCalls, 0]), stateShape),
      output_states_2: new Tensor("float32", new Float32Array([0, this.decoderCalls]), stateShape),
    };
  }
}

const BASE_SESSIONS = {
  encoder_model: {
    inputNames: ["input_features"],
    outputNames: ["outputs"],
  },
  decoder_model_merged: {
    inputNames: ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
    outputNames: ["outputs", "output_states_1", "output_states_2"],
  },
};

const BASE_CONFIG = {
  model_type: "nemo-conformer-tdt",
  "transformers.js_config": {
    transducer: {
      blank_token_id: 0,
      max_symbols_per_step: 2,
      subsampling_factor: 4,
      frame_shift_s: 0.01,
      vocab_size: 3,
      duration_start_index: 3,
      encoder_output_layout: "BDT",
      encoder_frame_layout: "BD1",
      decoder: {
        num_layers: 1,
        hidden_size: 2,
      },
    },
  },
};

export default () => {
  describe("NemoConformerForTDT", () => {
    it(
      "greedily decodes scripted token and duration logits",
      async () => {
        const tokenizer = {
          decode(ids) {
            const idArray = Array.isArray(ids) ? ids : [ids];
            return idArray
              .map((id) => {
                if (id === 1 || id === 1n) return " hello";
                if (id === 2 || id === 2n) return " world";
                return "";
              })
              .join("");
          },
        };

        const model = new MockNemoConformerForTDT(BASE_CONFIG, BASE_SESSIONS, [
          // step 1: emit token=1, duration=0
          { logits: [0.1, 10.0, 0.0, 8.0, 1.0, 0.5] },
          // step 2: emit blank, duration=1 -> move to next frame
          { logits: [9.0, 0.0, 0.0, 0.0, 8.0, 0.0] },
          // step 3: emit token=2, duration=2 -> jump to end
          { logits: [0.0, 0.0, 10.0, 0.0, 0.0, 9.0] },
        ]);

        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        const output = await model.transcribe(inputs, {
          tokenizer,
          return_timestamps: true,
          return_words: true,
          return_tokens: true,
        });

        expect(output.text).toBe("hello world");
        expect(output.utterance_timestamp).toEqual([0, 0.12]);
        expect(output.words).toEqual([expect.objectContaining({ text: "hello", start_time: 0, end_time: 0.04 }), expect.objectContaining({ text: "world", start_time: 0.04, end_time: 0.12 })]);
        expect(output.tokens).toEqual([expect.objectContaining({ id: 1, start_time: 0, end_time: 0.04 }), expect.objectContaining({ id: 2, start_time: 0.04, end_time: 0.12 })]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("fails fast when transducer config is missing", () => {
      const invalidConfig = { model_type: "nemo-conformer-tdt" };
      expect(() => new NemoConformerForTDT(invalidConfig, BASE_SESSIONS, {})).toThrow("Missing `transformers.js_config.transducer`");
    });

    it("requires explicit encoder_output_layout in transducer config", () => {
      const invalidConfig = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            encoder_output_layout: undefined,
          },
        },
      };
      expect(() => new NemoConformerForTDT(invalidConfig, BASE_SESSIONS, {})).toThrow("encoder_output_layout");
    });
  });

  describe("Nemo Conformer TDT utilities", () => {
    it(
      "computes delta and delta-delta features",
      async () => {
        const input = new Tensor(
          "float32",
          Float32Array.from([
            // T=4, F=2
            1, 2, 2, 4, 3, 6, 4, 8,
          ]),
          [1, 4, 2],
        );

        const split = computeTemporalDeltas(input, { order: 2, window: 1, concatenate: false });
        expect(split.delta.dims).toEqual([1, 4, 2]);
        expect(split.delta_delta.dims).toEqual([1, 4, 2]);

        const concatOrder1 = computeTemporalDeltas(input, { order: 1, window: 1, concatenate: true });
        expect(concatOrder1.dims).toEqual([1, 4, 4]);
        expect(Array.from(concatOrder1.data.slice(0, 8))).toEqual([
          1,
          2,
          0.5,
          1, // t0: base + delta
          2,
          4,
          1,
          2, // t1: base + delta
        ]);

        const concat = computeTemporalDeltas(input, { order: 2, window: 1, concatenate: true });
        expect(concat.dims).toEqual([1, 4, 6]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("rejects non-float32 tensors for temporal deltas", () => {
      const input = new Tensor("float64", Float64Array.from([1, 2, 2, 4]), [1, 2, 2]);
      expect(() => computeTemporalDeltas(input, { order: 1, window: 1, concatenate: true })).toThrow('type "float32"');
    });

    it(
      "creates stable audio cache keys",
      async () => {
        const a = Float32Array.from([0, 0.1, 0.2, 0.3]);
        const b = Float32Array.from([0, 0.1, 0.2, 0.4]);
        const ka1 = createAudioCacheKey(a, 16000);
        const ka2 = createAudioCacheKey(a, 16000);
        const kb = createAudioCacheKey(b, 16000);

        expect(ka1).toEqual(ka2);
        expect(ka1).not.toEqual(kb);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "evicts least-recently-used entries when full",
      async () => {
        const cache = new FeatureLRUCache({ max_entries: 2, max_size_mb: 4 });
        cache.set("a", new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]));
        cache.set("b", new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]));
        expect(cache.get("a")).not.toBeNull();

        cache.set("c", new Tensor("float32", new Float32Array([7, 8, 9]), [1, 3]));
        // `b` should be evicted because `a` was recently accessed.
        expect(cache.get("b")).toBeNull();
        expect(cache.get("a")).not.toBeNull();
        expect(cache.get("c")).not.toBeNull();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
