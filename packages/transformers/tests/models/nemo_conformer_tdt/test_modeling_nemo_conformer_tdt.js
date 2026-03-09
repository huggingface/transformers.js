import { NemoConformerForTDT, Tensor } from "../../../src/transformers.js";
import { createAudioCacheKey, FeatureLRUCache } from "../../../src/models/nemo_conformer_tdt/transducer_cache.js";
import { computeTemporalDeltas } from "../../../src/models/nemo_conformer_tdt/transducer_deltas.js";
import { buildNemoSegmentChunks, partitionNemoWordsIntoSegments, shouldEndSentenceAfterWord } from "../../../src/models/nemo_conformer_tdt/transducer_segment_offsets.js";
import { buildTransducerDetailedOutputs } from "../../../src/models/nemo_conformer_tdt/transducer_text.js";
import { buildTransducerWordOffsets } from "../../../src/models/nemo_conformer_tdt/transducer_word_offsets.js";
import { dedupeMergedWords } from "../../../src/models/nemo_conformer_tdt/transducer_window_merge.js";
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
      "throws on invalid runtime config: vocab_size must be > 0",
      async () => {
        const invalidConfig = {
          ...BASE_CONFIG,
          "transformers.js_config": {
            ...BASE_CONFIG["transformers.js_config"],
            transducer: {
              ...BASE_CONFIG["transformers.js_config"].transducer,
              vocab_size: 0,
            },
          },
        };
        const model = new MockNemoConformerForTDT(invalidConfig, BASE_SESSIONS, []);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: {
              decode: () => "",
              get_vocab: () => new Map([["a", 0]]),
            },
          }),
        ).rejects.toThrow("vocab_size");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "throws on invalid runtime config: blank_token_id must be < vocab_size",
      async () => {
        const invalidConfig = {
          ...BASE_CONFIG,
          "transformers.js_config": {
            ...BASE_CONFIG["transformers.js_config"],
            transducer: {
              ...BASE_CONFIG["transformers.js_config"].transducer,
              blank_token_id: 3,
            },
          },
        };
        const model = new MockNemoConformerForTDT(invalidConfig, BASE_SESSIONS, []);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: {
              decode: () => "",
              get_vocab: () =>
                new Map([
                  ["a", 0],
                  ["b", 1],
                  ["c", 2],
                ]),
            },
          }),
        ).rejects.toThrow("blank_token_id");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "throws on invalid runtime config: duration_start_index must be >= vocab_size",
      async () => {
        const invalidConfig = {
          ...BASE_CONFIG,
          "transformers.js_config": {
            ...BASE_CONFIG["transformers.js_config"],
            transducer: {
              ...BASE_CONFIG["transformers.js_config"].transducer,
              duration_start_index: 2,
            },
          },
        };
        const model = new MockNemoConformerForTDT(invalidConfig, BASE_SESSIONS, []);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: {
              decode: () => "",
              get_vocab: () =>
                new Map([
                  ["a", 0],
                  ["b", 1],
                  ["c", 2],
                ]),
            },
          }),
        ).rejects.toThrow("duration_start_index");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "throws explicit vocab resolution error when tokenizer.get_vocab returns a non-object",
      async () => {
        const configWithoutVocab = {
          ...BASE_CONFIG,
          "transformers.js_config": {
            ...BASE_CONFIG["transformers.js_config"],
            transducer: {
              ...BASE_CONFIG["transformers.js_config"].transducer,
              vocab_size: undefined,
            },
          },
        };
        const model = new MockNemoConformerForTDT(configWithoutVocab, BASE_SESSIONS, []);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: {
              decode: () => "",
              get_vocab: () => null,
            },
          }),
        ).rejects.toThrow("Unable to resolve vocabulary size");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("resolves vocab size from array tokenizers when config vocab_size is not set", () => {
      const configWithoutVocab = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            vocab_size: undefined,
          },
        },
      };
      const model = new MockNemoConformerForTDT(configWithoutVocab, BASE_SESSIONS, []);
      expect(
        model._resolveVocabSize({
          get_vocab: () => ["<blank>", "hello", "world"],
        }),
      ).toBe(3);
    });

    it("resolves vocab size from the maximum sparse tokenizer id when config vocab_size is not set", () => {
      const configWithoutVocab = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            vocab_size: undefined,
          },
        },
      };
      const model = new MockNemoConformerForTDT(configWithoutVocab, BASE_SESSIONS, []);
      expect(
        model._resolveVocabSize({
          get_vocab: () => ({
            "<blank>": 0,
            hello: 2,
            world: 7,
          }),
        }),
      ).toBe(8);
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
          returnTimestamps: true,
          returnWords: true,
          returnTokens: true,
        });

        expect(output.text).toBe("hello world");
        expect(output.isFinal).toBe(true);
        expect(output.utteranceTimestamp).toEqual([0, 0.12]);
        expect(output.words).toEqual([expect.objectContaining({ text: "hello", startTime: 0, endTime: 0.04 }), expect.objectContaining({ text: "world", startTime: 0.04, endTime: 0.12 })]);
        expect(output.tokens).toEqual([expect.objectContaining({ id: 1, startTime: 0, endTime: 0.04 }), expect.objectContaining({ id: 2, startTime: 0.04, endTime: 0.12 })]);
        expect(output.confidence).toEqual(expect.objectContaining({ utterance: expect.any(Number), wordAverage: expect.any(Number), averageLogProb: expect.any(Number) }));
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
          returnTimestamps: true,
          returnTokens: true,
        });

        expect(output.tokens).toHaveLength(1);
        expect(output.tokens[0]).toEqual(expect.objectContaining({ startTime: 0, endTime: 0.12 }));
        expect(output.utteranceTimestamp).toEqual([0, 0.12]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "aggregates frame confidences per encoder frame (not per decode step)",
      async () => {
        const model = new MockNemoConformerForTDT(BASE_CONFIG, BASE_SESSIONS, [
          // Frame 0: emit token=1, step=0
          { logits: [0.0, 4.0, -2.0, 9.0, 1.0, 0.0] },
          // Frame 0: emit token=2, step=0 (hits max_symbols_per_step and advances frame)
          { logits: [0.0, -1.0, 3.0, 9.0, 1.0, 0.0] },
          // Frame 1: emit blank, step=2 -> exits decode loop
          { logits: [5.0, 0.0, 0.0, 0.0, 1.0, 9.0] },
        ]);

        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        const output = await model.transcribe(inputs, {
          returnTimestamps: false,
          returnFrameConfidences: true,
        });

        expect(output.confidence.frames).toHaveLength(2);
        expect(output.confidence.frames[0]).toBeCloseTo(0.9579343795, 6);
        expect(output.confidence.frameAverage).toBeCloseTo((output.confidence.frames[0] + output.confidence.frames[1]) / 2, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "rejects non-finite timeOffset",
      async () => {
        const model = new MockNemoConformerForTDT(BASE_CONFIG, BASE_SESSIONS, [{ logits: [9.0, 0.0, 0.0, 1.0] }]);
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        await expect(
          model.transcribe(inputs, {
            tokenizer: { decode: () => "" },
            returnTimestamps: true,
            timeOffset: Number.NaN,
          }),
        ).rejects.toThrow("timeOffset");
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
            returnTimestamps: false,
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

    it("rejects invalid encoder_input_layout at construction time", () => {
      const invalidConfig = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            encoder_input_layout: "BAD",
          },
        },
      };
      expect(() => new NemoConformerForTDT(invalidConfig, BASE_SESSIONS, {})).toThrow("encoder_input_layout");
    });

    it("applies encoder_input_layout to canonical input_features feeds", () => {
      const config = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            encoder_input_layout: "BFT",
          },
        },
      };
      const model = new NemoConformerForTDT(config, BASE_SESSIONS, {});
      const input_features = new Tensor("float32", new Float32Array([1, 2, 3, 4, 5, 6]), [1, 3, 2]);

      const { feeds, disposables } = model._buildEncoderFeeds({ input_features });

      try {
        expect(disposables).toHaveLength(1);
        expect(feeds.input_features).not.toBe(input_features);
        expect(feeds.input_features.dims).toEqual([1, 2, 3]);
        expect(Array.from(feeds.input_features.data)).toEqual([1, 3, 5, 2, 4, 6]);
      } finally {
        for (const tensor of disposables) {
          tensor.dispose();
        }
        input_features.dispose();
      }
    });

    it("rejects invalid encoder_frame_layout at construction time", () => {
      const invalidConfig = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            encoder_frame_layout: "BAD",
          },
        },
      };
      expect(() => new NemoConformerForTDT(invalidConfig, BASE_SESSIONS, {})).toThrow("encoder_frame_layout");
    });

    it(
      "fails fast when named encoder output is missing at runtime",
      async () => {
        class MissingEncoderOutputModel extends NemoConformerForTDT {
          async _runEncoder() {
            return {
              outputs: new Tensor("float32", new Float32Array([0.1, 0.2]), [1, 2, 1]),
            };
          }

          async _runDecoder() {
            const stateShape = [1, 1, 2];
            return {
              outputs: new Tensor("float32", new Float32Array([9.0, 0.0, 0.0, 8.0]), [1, 1, 4]),
              output_states_1: new Tensor("float32", new Float32Array([0, 0]), stateShape),
              output_states_2: new Tensor("float32", new Float32Array([0, 0]), stateShape),
            };
          }
        }

        const config = {
          ...BASE_CONFIG,
          "transformers.js_config": {
            ...BASE_CONFIG["transformers.js_config"],
            transducer: {
              ...BASE_CONFIG["transformers.js_config"].transducer,
              io: { encoder_output: "encoder_out" },
            },
          },
        };
        const sessions = {
          ...BASE_SESSIONS,
          encoder_model: {
            ...BASE_SESSIONS.encoder_model,
            outputNames: ["encoder_out"],
          },
        };
        const model = new MissingEncoderOutputModel(config, sessions, {});
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        await expect(model.transcribe(inputs, { tokenizer: { decode: () => "" } })).rejects.toThrow('encoder output "encoder_out" was not returned');
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "fails fast when named decoder logits output is missing at runtime",
      async () => {
        class MissingDecoderOutputModel extends NemoConformerForTDT {
          async _runEncoder() {
            return {
              outputs: new Tensor("float32", new Float32Array([0.1, 0.2]), [1, 2, 1]),
            };
          }

          async _runDecoder() {
            const stateShape = [1, 1, 2];
            return {
              unexpected_logits: new Tensor("float32", new Float32Array([9.0, 0.0, 0.0, 8.0]), [1, 1, 4]),
              output_states_1: new Tensor("float32", new Float32Array([0, 0]), stateShape),
              output_states_2: new Tensor("float32", new Float32Array([0, 0]), stateShape),
            };
          }
        }

        const model = new MissingDecoderOutputModel(BASE_CONFIG, BASE_SESSIONS, {});
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        await expect(model.transcribe(inputs, { tokenizer: { decode: () => "" } })).rejects.toThrow('decoder output "outputs" was not returned');
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "fails fast when named decoder state outputs are missing at runtime",
      async () => {
        class MissingDecoderStateOutputsModel extends NemoConformerForTDT {
          async _runEncoder() {
            return {
              outputs: new Tensor("float32", new Float32Array([0.1, 0.2]), [1, 2, 1]),
            };
          }

          async _runDecoder() {
            return {
              outputs: new Tensor("float32", new Float32Array([9.0, 0.0, 0.0, 8.0]), [1, 1, 4]),
            };
          }
        }

        const model = new MissingDecoderStateOutputsModel(BASE_CONFIG, BASE_SESSIONS, {});
        const inputs = {
          input_features: new Tensor("float32", new Float32Array([0, 0, 0, 0, 0, 0]), [1, 3, 2]),
        };

        await expect(model.transcribe(inputs, { tokenizer: { decode: () => "" } })).rejects.toThrow('decoder state outputs "output_states_1" and "output_states_2" were not returned');
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("rejects duplicate decoder output aliases in transducer io config", () => {
      const invalidConfig = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            io: {
              decoder_output: "outputs",
              decoder_output_state_1: "outputs",
              decoder_output_state_2: "output_states_2",
            },
          },
        },
      };
      expect(() => new NemoConformerForTDT(invalidConfig, BASE_SESSIONS, {})).toThrow("must be distinct");
    });

    it("rejects duplicate decoder input aliases in transducer io config", () => {
      const invalidConfig = {
        ...BASE_CONFIG,
        "transformers.js_config": {
          ...BASE_CONFIG["transformers.js_config"],
          transducer: {
            ...BASE_CONFIG["transformers.js_config"].transducer,
            io: {
              decoder_encoder: "encoder_outputs",
              decoder_token: "targets",
              decoder_token_length: "target_length",
              decoder_state_1: "input_states_1",
              decoder_state_2: "input_states_1",
            },
          },
        },
      };
      expect(() => new NemoConformerForTDT(invalidConfig, BASE_SESSIONS, {})).toThrow("must be distinct");
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

        const output = await model.transcribe(inputs, { returnTimestamps: false });
        expect(output).toEqual(expect.objectContaining({ text: "" }));
        expect(model.auxDisposals).toBe(1);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Nemo Conformer TDT utilities", () => {
    it("uses conservative sentence boundaries for punctuation, abbreviations, and long silences", () => {
      expect(shouldEndSentenceAfterWord({ text: "Hello." }, { text: "World" }, 0)).toBe(true);
      expect(shouldEndSentenceAfterWord({ text: "U.S." }, { text: "Report" }, 0)).toBe(false);
      expect(shouldEndSentenceAfterWord({ text: "3." }, { text: "Title" }, 0)).toBe(false);
      expect(shouldEndSentenceAfterWord({ text: "I." }, { text: "Overview" }, 0)).toBe(false);
      expect(shouldEndSentenceAfterWord({ text: "Dr." }, { text: "Brown" }, 0)).toBe(false);
      expect(shouldEndSentenceAfterWord({ text: "wait" }, { text: "Next" }, 3.2)).toBe(true);
    });

    it("partitions timed words into conservative sentence-like chunks", () => {
      const words = [
        { text: "Hello.", startTime: 0, endTime: 0.4 },
        { text: "World", startTime: 0.5, endTime: 0.8 },
        { text: "again.", startTime: 0.8, endTime: 1.1 },
        { text: "U.S.", startTime: 1.2, endTime: 1.5 },
        { text: "Report", startTime: 1.6, endTime: 2.0 },
        { text: "update.", startTime: 2.0, endTime: 2.4 },
        { text: "pause", startTime: 6.0, endTime: 6.3 },
        { text: "Next", startTime: 9.5, endTime: 9.8 },
        { text: "sentence.", startTime: 9.8, endTime: 10.2 },
      ];

      const segments = partitionNemoWordsIntoSegments(words);
      expect(segments.map((x) => x.text)).toEqual(["Hello.", "World again.", "U.S. Report update.", "pause", "Next sentence."]);
      expect(segments.map((x) => x.timestamp)).toEqual([
        [0, 0.4],
        [0.5, 1.1],
        [1.2, 2.4],
        [6.0, 6.3],
        [9.5, 10.2],
      ]);
      expect(buildNemoSegmentChunks(words)).toEqual([
        { text: "Hello.", timestamp: [0, 0.4] },
        { text: "World again.", timestamp: [0.5, 1.1] },
        { text: "U.S. Report update.", timestamp: [1.2, 2.4] },
        { text: "pause", timestamp: [6.0, 6.3] },
        { text: "Next sentence.", timestamp: [9.5, 10.2] },
      ]);
    });

    it("keeps word boundaries from the final decoded text for numeric and punctuation tokens", () => {
      const rawById = {
        1: "▁score",
        2: ".",
        3: "48",
        4: "-",
        5: "year",
        6: "-",
        7: "old",
        8: "▁with",
        9: "0",
        10: ".",
        11: "5",
      };
      const tokenizer = {
        get_vocab() {
          return rawById;
        },
        decode(ids) {
          if (ids.length === 1) {
            return rawById[ids[0]].replace(/^▁/, "");
          }
          return "score. 48-year-old with 0.5";
        },
      };

      const output = buildTransducerDetailedOutputs(
        tokenizer,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [
          [0.0, 0.3],
          [0.3, 0.4],
          [0.5, 0.8],
          [0.8, 0.85],
          [0.85, 1.05],
          [1.05, 1.1],
          [1.1, 1.3],
          [1.4, 1.7],
          [1.8, 1.9],
          [1.9, 1.95],
          [1.95, 2.05],
        ],
      );

      expect(output.words.map((x) => x.text)).toEqual(["score.", "48-year-old", "with", "0.5"]);
      expect(output.tokens.map((x) => x.token)).toEqual(["score", ".", "48", "-", "year", "-", "old", "with", "0", ".", "5"]);
    });

    it("does not collapse distinct overlapping punctuation-only tokens during merge dedupe", () => {
      expect(
        dedupeMergedWords([
          { text: ".", startTime: 1.0, endTime: 1.3 },
          { text: "?", startTime: 1.2, endTime: 1.5 },
          { text: "?", startTime: 1.2, endTime: 1.6 },
        ]),
      ).toEqual([
        { text: ".", startTime: 1.0, endTime: 1.3 },
        { text: "?", startTime: 1.2, endTime: 1.6 },
      ]);
    });

    it("builds word offsets from array-backed tokenizer vocabularies", () => {
      const vocab = ["<blank>", "▁hello", "▁world"];
      const tokenizer = {
        get_vocab() {
          return vocab;
        },
        decode(ids) {
          const pieces = ids.map((id) => vocab[id] ?? "").join("");
          return pieces.replace(/▁/g, "").trim();
        },
      };

      const output = buildTransducerWordOffsets(
        tokenizer,
        [1, 2],
        [
          [0.0, 0.3],
          [0.3, 0.6],
        ],
        null,
        "hello world",
      );

      expect(output.words.map((x) => x.text)).toEqual(["hello", "world"]);
      expect(output.tokens.map((x) => x.rawToken)).toEqual(["▁hello", "▁world"]);
      expect(output.tokens.map((x) => x.isWordStart)).toEqual([true, true]);
    });

    it("falls back to decoded token text when tokenizer vocab metadata is unavailable", () => {
      const token_ids = [1, 2];
      const timestamps = [
        [0.0, 0.3],
        [0.3, 0.6],
      ];

      const fromNull = buildTransducerWordOffsets(
        {
          get_vocab: () => null,
          decode(ids) {
            return ids[0] === 1 ? " hello" : "world";
          },
        },
        token_ids,
        timestamps,
        null,
        "hello world",
      );
      const fromPrimitive = buildTransducerWordOffsets(
        {
          get_vocab: () => 42,
          decode(ids) {
            return ids[0] === 1 ? " hello" : "world";
          },
        },
        token_ids,
        timestamps,
        null,
        "hello world",
      );

      expect(fromNull.words.map((x) => x.text)).toEqual(["hello", "world"]);
      expect(fromPrimitive.words.map((x) => x.text)).toEqual(["hello", "world"]);
    });

    it("rejects mismatched empty timestamp inputs for word offsets", () => {
      expect(() =>
        buildTransducerWordOffsets(
          {
            decode: () => "hello",
          },
          [1],
          [],
        ),
      ).toThrow("equal lengths");
    });

    it("requires a tokenizer for non-empty word offsets", () => {
      expect(() => buildTransducerWordOffsets(null, [1], [[0.0, 0.3]], null, "hello")).toThrow("requires a tokenizer");
    });

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
      expect(files).toEqual(["config.json", "onnx/encoder_model_fp16.onnx", "onnx/decoder_model_merged_q4.onnx"]);
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
        expect(cache.set("a", new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]))).toBe(true);
        expect(cache.set("b", new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]))).toBe(true);
        expect(cache.get("a")).not.toBeNull();

        expect(cache.set("c", new Tensor("float32", new Float32Array([7, 8, 9]), [1, 3]))).toBe(true);
        // `b` should be evicted because `a` was recently accessed.
        expect(cache.get("b")).toBeNull();
        expect(cache.get("a")).not.toBeNull();
        expect(cache.get("c")).not.toBeNull();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("disposes replaced cache entries", () => {
      const cache = new FeatureLRUCache({ max_entries: 4, max_size_mb: 4 });
      const originalDispose = Tensor.prototype.dispose;
      let disposeCalls = 0;
      Tensor.prototype.dispose = function () {
        disposeCalls += 1;
        return originalDispose.call(this);
      };

      try {
        cache.set("x", new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]));
        cache.set("x", new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]));
        expect(disposeCalls).toBe(1);
      } finally {
        Tensor.prototype.dispose = originalDispose;
        cache.clear();
      }
    });

    it("does not dispose when re-setting the same value object for an existing key", () => {
      const cache = new FeatureLRUCache({ max_entries: 4, max_size_mb: 4 });
      const tensor = new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]);
      let disposeCalls = 0;
      const originalDispose = tensor.dispose.bind(tensor);
      tensor.dispose = () => {
        disposeCalls += 1;
        originalDispose();
      };

      cache.set("x", tensor);
      cache.set("x", tensor);
      expect(cache.get("x")).toBe(tensor);
      expect(disposeCalls).toBe(0);

      cache.clear();
      expect(disposeCalls).toBe(1);
    });

    it("disposes tensors on eviction and clear without double-disposing shared refs", () => {
      const cache = new FeatureLRUCache({ max_entries: 1, max_size_mb: 4 });
      const originalDispose = Tensor.prototype.dispose;
      let disposeCalls = 0;
      Tensor.prototype.dispose = function () {
        disposeCalls += 1;
        return originalDispose.call(this);
      };

      try {
        const sharedA = new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]);
        cache.set("a", {
          input_features: sharedA,
          attention_mask: sharedA,
        });
        const sharedB = new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]);
        cache.set("b", {
          input_features: sharedB,
          attention_mask: sharedB,
        });
        // Eviction of "a" should dispose sharedA once, despite duplicate field references.
        expect(disposeCalls).toBe(1);

        cache.clear();
        // Clear should dispose sharedB once.
        expect(disposeCalls).toBe(2);
      } finally {
        Tensor.prototype.dispose = originalDispose;
      }
    });

    it("defers disposal for borrowed cache entries until they are released", () => {
      const cache = new FeatureLRUCache({ max_entries: 1, max_size_mb: 4 });
      const tensorA = new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]);
      const tensorB = new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]);
      let disposeCalls = 0;
      const track = (tensor) => {
        const originalDispose = tensor.dispose.bind(tensor);
        tensor.dispose = () => {
          disposeCalls += 1;
          originalDispose();
        };
      };
      track(tensorA);
      track(tensorB);

      cache.set("a", tensorA);
      const borrowedA = cache.acquire("a");
      expect(borrowedA?.value).toBe(tensorA);

      cache.set("b", tensorB);
      expect(disposeCalls).toBe(0);
      borrowedA?.release();
      expect(disposeCalls).toBe(1);

      const borrowedB = cache.acquire("b");
      expect(borrowedB?.value).toBe(tensorB);
      cache.clear();
      expect(disposeCalls).toBe(1);
      borrowedB?.release();
      expect(disposeCalls).toBe(2);
    });

    it("keeps borrowed entry bytes counted until release", () => {
      const cache = new FeatureLRUCache({ max_entries: 4, max_size_mb: 0.00002 });
      const tensorA = new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]);
      const tensorB = new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]);

      let tensorADisposals = 0;
      const disposeA = tensorA.dispose.bind(tensorA);
      tensorA.dispose = () => {
        tensorADisposals += 1;
        disposeA();
      };

      let tensorBDisposals = 0;
      const disposeB = tensorB.dispose.bind(tensorB);
      tensorB.dispose = () => {
        tensorBDisposals += 1;
        disposeB();
      };

      expect(cache.set("a", tensorA)).toBe(true);
      const borrowedA = cache.acquire("a");
      expect(borrowedA?.value).toBe(tensorA);

      expect(cache.set("b", tensorB)).toBe(false);
      expect(cache.get("a")).toBeNull();
      expect(cache.get("b")).toBeNull();
      expect(cache.stats().entries).toBe(0);
      expect(cache.stats().size_mb).toBeGreaterThan(0);
      expect(tensorADisposals).toBe(0);
      expect(tensorBDisposals).toBe(1);

      borrowedA?.release();
      expect(cache.stats().size_mb).toBe(0);
      expect(tensorADisposals).toBe(1);
    });

    it("treats zero cache limits as explicit no-cache mode without disposing inserted values", () => {
      const byEntries = new FeatureLRUCache({ max_entries: 0, max_size_mb: 4 });
      const bySize = new FeatureLRUCache({ max_entries: 4, max_size_mb: 0 });
      const t1 = new Tensor("float32", new Float32Array([1, 2, 3]), [1, 3]);
      const t2 = new Tensor("float32", new Float32Array([4, 5, 6]), [1, 3]);

      let t1Disposals = 0;
      const t1Dispose = t1.dispose.bind(t1);
      t1.dispose = () => {
        t1Disposals += 1;
        t1Dispose();
      };
      let t2Disposals = 0;
      const t2Dispose = t2.dispose.bind(t2);
      t2.dispose = () => {
        t2Disposals += 1;
        t2Dispose();
      };

      expect(byEntries.set("x", t1)).toBe(false);
      expect(bySize.set("y", t2)).toBe(false);
      expect(byEntries.get("x")).toBeNull();
      expect(bySize.get("y")).toBeNull();
      expect(t1Disposals).toBe(0);
      expect(t2Disposals).toBe(0);

      t1.dispose();
      t2.dispose();
      expect(t1Disposals).toBe(1);
      expect(t2Disposals).toBe(1);
    });

    it("skips caching oversized values without disposing caller-owned tensors", () => {
      const cache = new FeatureLRUCache({ max_entries: 4, max_size_mb: 0.000001 });
      const tensor = new Tensor("float32", new Float32Array([1, 2]), [1, 2]);
      let disposeCalls = 0;
      const originalDispose = tensor.dispose.bind(tensor);
      tensor.dispose = () => {
        disposeCalls += 1;
        originalDispose();
      };

      expect(cache.set("big", tensor)).toBe(false);
      expect(cache.get("big")).toBeNull();
      expect(disposeCalls).toBe(0);

      tensor.dispose();
      expect(disposeCalls).toBe(1);
    });

    it("ignores non-numeric byteLength values in size estimation", () => {
      const cache = new FeatureLRUCache({ max_entries: 4, max_size_mb: 4 });
      cache.set("x", { byteLength: "invalid" });
      expect(cache.stats().entries).toBe(1);
      expect(cache.stats().size_mb).toBe(0);
      cache.clear();
    });

    it("rejects invalid cache limits", () => {
      expect(() => new FeatureLRUCache({ max_entries: -1 })).toThrow("max_entries");
      expect(() => new FeatureLRUCache({ max_entries: 1.25 })).toThrow("max_entries");
      expect(() => new FeatureLRUCache({ max_size_mb: -1 })).toThrow("max_size_mb");
      expect(() => new FeatureLRUCache({ max_size_mb: Number.POSITIVE_INFINITY })).toThrow("max_size_mb");
    });
  });
};
