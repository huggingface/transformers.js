import { NemoConformerForTDT, Tensor } from "../../../src/transformers.js";
import { createAudioCacheKey, FeatureLRUCache } from "../../../src/models/nemo_conformer_tdt/transducer_cache.js";
import { computeTemporalDeltas } from "../../../src/models/nemo_conformer_tdt/transducer_deltas.js";
import { MODEL_TYPE_MAPPING, MODEL_TYPES } from "../../../src/models/modeling_utils.js";
import { get_model_files } from "../../../src/utils/model_registry/get_model_files.js";

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
    it("maps NemoConformerForTDT to MODEL_TYPES.NemoConformerTDT", () => {
      expect(MODEL_TYPE_MAPPING.get("NemoConformerForTDT")).toBe(MODEL_TYPES.NemoConformerTDT);
      expect(MODEL_TYPE_MAPPING.get("nemo-conformer-tdt")).toBe(MODEL_TYPES.NemoConformerTDT);
    });

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

    it(
      "clamps token timestamps when step jumps beyond remaining frames",
      async () => {
        const tokenizer = {
          decode(ids) {
            const idArray = Array.isArray(ids) ? ids : [ids];
            return idArray.map((id) => (id === 1 || id === 1n ? " token" : "")).join("");
          },
        };

        const model = new MockNemoConformerForTDT(BASE_CONFIG, BASE_SESSIONS, [
          // Emit token=1 with duration index choosing a large step (argmax at tail).
          { logits: [0.1, 10.0, 0.0, 0.0, 0.0, 0.0, 12.0] },
        ]);

        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        const output = await model.transcribe(inputs, {
          tokenizer,
          return_timestamps: true,
          return_tokens: true,
        });

        expect(output.tokens).toHaveLength(1);
        expect(output.tokens[0]).toEqual(expect.objectContaining({ start_time: 0, end_time: 0.12 }));
        expect(output.utterance_timestamp).toEqual([0, 0.12]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "fails fast when duration logits are required but missing",
      async () => {
        const model = new MockNemoConformerForTDT(BASE_CONFIG, BASE_SESSIONS, [
          // Only vocab logits are returned; duration head is missing.
          { logits: [0.1, 10.0, 0.0] },
        ]);

        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: { decode: () => "" },
            return_timestamps: false,
          }),
        ).rejects.toThrow("missing duration logits");
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

    it(
      "disposes encoder outputs when frame-count validation fails before decode",
      async () => {
        class BadEncoderOutputModel extends NemoConformerForTDT {
          constructor(config, sessions, encoderOutput) {
            super(config, sessions, {});
            this.encoderOutput = encoderOutput;
          }

          async _runEncoder() {
            return { outputs: this.encoderOutput };
          }
        }

        const badEncoderOutput = new Tensor("float32", new Float32Array([0, 1, 2, 3]), [2, 2]);
        let disposed = 0;
        const originalDispose = badEncoderOutput.dispose.bind(badEncoderOutput);
        badEncoderOutput.dispose = () => {
          disposed += 1;
          originalDispose();
        };

        const model = new BadEncoderOutputModel(BASE_CONFIG, BASE_SESSIONS, badEncoderOutput);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: { decode: () => "" },
          }),
        ).rejects.toThrow("expected encoder output dims");
        expect(disposed).toBe(1);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "disposes auxiliary decoder tensor outputs per decode step",
      async () => {
        class AuxDecoderOutputModel extends NemoConformerForTDT {
          constructor(config, sessions) {
            super(config, sessions, {});
            this.auxDisposals = 0;
          }

          async _runEncoder() {
            return {
              outputs: new Tensor("float32", new Float32Array([0.1, 0.2]), [1, 2, 1]),
            };
          }

          async _runDecoder() {
            const stateShape = [1, 1, 2];
            const aux = new Tensor("float32", new Float32Array([1, 2, 3]), [1, 1, 3]);
            const originalDispose = aux.dispose.bind(aux);
            aux.dispose = () => {
              this.auxDisposals += 1;
              originalDispose();
            };
            return {
              outputs: new Tensor("float32", new Float32Array([10.0, 0.0, 0.0, 8.0, 0.0]), [1, 1, 5]),
              output_states_1: new Tensor("float32", new Float32Array([0, 0]), stateShape),
              output_states_2: new Tensor("float32", new Float32Array([0, 0]), stateShape),
              auxiliary_scores: aux,
            };
          }
        }

        const model = new AuxDecoderOutputModel(BASE_CONFIG, BASE_SESSIONS);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]),
        };

        const output = await model.transcribe(inputs, { return_timestamps: false });
        expect(output).toEqual(expect.objectContaining({ text: "" }));
        expect(model.auxDisposals).toBe(1);
      },
      MAX_TEST_EXECUTION_TIME,
    );
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

    it("disposes intermediate delta tensors in concatenate paths", () => {
      const input = new Tensor("float32", Float32Array.from([1, 2, 2, 4, 3, 6, 4, 8]), [1, 4, 2]);
      const originalDispose = Tensor.prototype.dispose;
      let disposeCalls = 0;
      Tensor.prototype.dispose = function () {
        disposeCalls += 1;
        return originalDispose.call(this);
      };

      try {
        const order1 = computeTemporalDeltas(input, { order: 1, window: 1, concatenate: true });
        const order2 = computeTemporalDeltas(input, { order: 2, window: 1, concatenate: true });
        expect(order1.dims).toEqual([1, 4, 4]);
        expect(order2.dims).toEqual([1, 4, 6]);
      } finally {
        Tensor.prototype.dispose = originalDispose;
      }

      // order=1 concat disposes one intermediate tensor, order=2 concat disposes two.
      expect(disposeCalls).toBe(3);
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

    it("uses Nemo encoder selector key when resolving model files", async () => {
      const files = await get_model_files("dummy/nemo", {
        local_files_only: true,
        config: {
          architectures: ["UnknownArch"],
          model_type: "nemo-conformer-tdt",
          "transformers.js_config": {},
        },
        dtype: {
          model: "int8",
          encoder_model: "fp16",
          decoder_model_merged: "q4",
        },
      });
      expect(files).toEqual([
        "config.json",
        "onnx/encoder_model_fp16.onnx",
        "onnx/decoder_model_merged_q4.onnx",
      ]);
    });

    it(
      "distinguishes long waveforms that differ at unsampled indices",
      async () => {
        const a = new Float32Array(10000);
        const b = new Float32Array(10000);
        b[1] = 0.12345; // Index 1 was skipped by the prior stride-based hash for this length.

        const ka = createAudioCacheKey(a, 16000);
        const kb = createAudioCacheKey(b, 16000);
        expect(ka).not.toEqual(kb);
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

    it("rejects invalid cache limits", () => {
      expect(() => new FeatureLRUCache({ max_entries: -1 })).toThrow("max_entries");
      expect(() => new FeatureLRUCache({ max_entries: 1.25 })).toThrow("max_entries");
      expect(() => new FeatureLRUCache({ max_size_mb: -1 })).toThrow("max_size_mb");
      expect(() => new FeatureLRUCache({ max_size_mb: Number.POSITIVE_INFINITY })).toThrow("max_size_mb");
    });
  });
};
