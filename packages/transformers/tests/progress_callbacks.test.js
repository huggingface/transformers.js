import { pipeline, LlamaForCausalLM, AutoModelForCausalLM, WhisperForConditionalGeneration, Gemma3ForConditionalGeneration, Gemma3nForConditionalGeneration, VoxtralRealtimeForConditionalGeneration } from "../src/transformers.js";

import { init, MAX_MODEL_LOAD_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "./init.js";

// Initialise the testing environment
init();

/**
 * Collects progress events during a loader call and returns them.
 * @param {(cb: Function) => Promise<{ dispose(): Promise<void> }>} loader
 * @returns {Promise<{ events: import('../src/utils/core.js').ProgressInfo[], dispose: () => Promise<void> }>}
 */
async function collectEvents(loader) {
  /** @type {import('../src/utils/core.js').ProgressInfo[]} */
  const events = [];
  const result = await loader((info) => events.push(info));
  return { events, dispose: () => result.dispose() };
}

/**
 * Validates progress_total events:
 *  1. loaded is monotonically non-decreasing
 *  2. total is constant across all events
 *  3. final progress value is 100
 * @param {Array<Object>} totalEvents
 */
function expectValidTotalEvents(totalEvents) {
  expect(totalEvents.length).toBeGreaterThan(0);

  for (const event of totalEvents) {
    expect(event).toHaveProperty("status", "progress_total");
    expect(event).toHaveProperty("progress");
    expect(event).toHaveProperty("loaded");
    expect(event).toHaveProperty("total");
    expect(event).toHaveProperty("files");
    expect(typeof event.progress).toBe("number");
    expect(event.progress).toBeGreaterThanOrEqual(0);
    expect(event.progress).toBeLessThanOrEqual(100);
    expect(event.loaded).toBeLessThanOrEqual(event.total);
  }

  // 1. loaded should be monotonically non-decreasing
  for (let i = 1; i < totalEvents.length; i++) {
    expect(totalEvents[i].loaded).toBeGreaterThanOrEqual(totalEvents[i - 1].loaded);
  }

  // 2. total should be constant across all events
  const expectedTotal = totalEvents[0].total;
  for (const event of totalEvents) {
    expect(event.total).toBe(expectedTotal);
  }

  // 3. final progress value should be 100
  expect(totalEvents.at(-1).progress).toBe(100);
  expect(totalEvents.at(-1).loaded).toBe(totalEvents.at(-1).total);
}

/**
 * Validates per-file event lifecycle and structure.
 * @param {Array<Object>} events All collected events.
 * @param {string} model_id Expected model name on events.
 * @param {string[]} expectedFiles File paths that must be present in the files map.
 */
function expectValidEventLifecycle(events, model_id, expectedFiles) {
  const totalEvents = events.filter((e) => e.status === "progress_total");
  expectValidTotalEvents(totalEvents);

  // Exact file count in the final progress_total
  const lastFiles = totalEvents.at(-1).files;
  expect(Object.keys(lastFiles).length).toBe(expectedFiles.length);

  // All expected files are present and fully loaded
  for (const file of expectedFiles) {
    expect(lastFiles).toHaveProperty([file]);
    expect(lastFiles[file].loaded).toBe(lastFiles[file].total);
  }

  // Every file emits initiate -> ... -> done lifecycle
  const trackedFiles = new Set(events.filter((e) => e.file).map((e) => e.file));
  for (const file of trackedFiles) {
    const fileEvents = events.filter((e) => e.file === file);
    expect(fileEvents[0].status).toBe("initiate");
    expect(fileEvents.at(-1).status).toBe("done");
  }

  // All events with a name field should reference the correct model
  for (const event of events) {
    if (event.name) {
      expect(event.name).toBe(model_id);
    }
  }

  // No double-wrapping: at most one progress_total per progress event
  const progressEvents = events.filter((e) => e.status === "progress");
  expect(totalEvents.length).toBeLessThanOrEqual(progressEvents.length);
}

describe("Progress Callbacks", () => {
  // ---- Llama (decoder-only) ----
  // from_pretrained files: config.json, onnx/model.onnx, generation_config.json
  // pipeline files: + tokenizer.json, tokenizer_config.json
  describe("Llama (decoder-only)", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    it(
      "pipeline('text-generation')",
      async () => {
        const { events, dispose } = await collectEvents((cb) => pipeline("text-generation", model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/model.onnx", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "LlamaForCausalLM.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => LlamaForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/model.onnx", "generation_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "AutoModelForCausalLM.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => AutoModelForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/model.onnx", "generation_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  // ---- Whisper (encoder-decoder) ----
  // from_pretrained files: config.json, onnx/encoder_model.onnx, onnx/decoder_model_merged.onnx, generation_config.json
  // pipeline files: + tokenizer.json, tokenizer_config.json, preprocessor_config.json
  describe("Whisper (encoder-decoder)", () => {
    const model_id = "onnx-internal-testing/tiny-random-WhisperForConditionalGeneration";

    it(
      "pipeline('automatic-speech-recognition')",
      async () => {
        const { events, dispose } = await collectEvents((cb) => pipeline("automatic-speech-recognition", model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "WhisperForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => WhisperForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "generation_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  // ---- Gemma3 (image-text-to-text) ----
  // from_pretrained files: config.json, onnx/embed_tokens.onnx, onnx/embed_tokens.onnx_data,
  //   onnx/decoder_model_merged.onnx, onnx/decoder_model_merged.onnx_data,
  //   onnx/vision_encoder.onnx, onnx/vision_encoder.onnx_data, generation_config.json
  describe("Gemma3 (image-text-to-text)", () => {
    const model_id = "onnx-internal-testing/tiny-random-Gemma3ForConditionalGeneration";

    it(
      "Gemma3ForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => Gemma3ForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/embed_tokens.onnx", "onnx/embed_tokens.onnx_data", "onnx/decoder_model_merged.onnx", "onnx/decoder_model_merged.onnx_data", "onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx_data", "generation_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  // ---- Gemma3n (image-audio-text-to-text) ----
  // from_pretrained files: config.json, onnx/embed_tokens.onnx, onnx/embed_tokens.onnx_data,
  //   onnx/decoder_model_merged.onnx, onnx/decoder_model_merged.onnx_data,
  //   onnx/audio_encoder.onnx, onnx/audio_encoder.onnx_data,
  //   onnx/vision_encoder.onnx, onnx/vision_encoder.onnx_data, generation_config.json
  describe("Gemma3n (image-audio-text-to-text)", () => {
    const model_id = "onnx-internal-testing/tiny-random-Gemma3nForConditionalGeneration";

    it(
      "Gemma3nForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => Gemma3nForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/embed_tokens.onnx", "onnx/embed_tokens.onnx_data", "onnx/decoder_model_merged.onnx", "onnx/decoder_model_merged.onnx_data", "onnx/audio_encoder.onnx", "onnx/audio_encoder.onnx_data", "onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx_data", "generation_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  // ---- VoxtralRealtime (audio-text-to-text) ----
  // from_pretrained files: config.json, onnx/embed_tokens.onnx, onnx/embed_tokens.onnx_data,
  //   onnx/decoder_model_merged.onnx, onnx/decoder_model_merged.onnx_data,
  //   onnx/audio_encoder.onnx, onnx/audio_encoder.onnx_data, generation_config.json
  describe("VoxtralRealtime (audio-text-to-text)", () => {
    const model_id = "onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration";

    it(
      "VoxtralRealtimeForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => VoxtralRealtimeForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        expectValidEventLifecycle(events, model_id, ["config.json", "onnx/embed_tokens.onnx", "onnx/embed_tokens.onnx_data", "onnx/decoder_model_merged.onnx", "onnx/decoder_model_merged.onnx_data", "onnx/audio_encoder.onnx", "onnx/audio_encoder.onnx_data", "generation_config.json"]);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  // ---- Edge cases ----
  describe("Edge cases", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    it(
      "no progress_total without progress_callback",
      async () => {
        // When no progress_callback is provided, nothing should throw
        const model = await LlamaForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
        await model.dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "per-file progress events have loaded <= total",
      async () => {
        const { events, dispose } = await collectEvents((cb) => LlamaForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const progressEvents = events.filter((e) => e.status === "progress");
        for (const event of progressEvents) {
          expect(event.loaded).toBeLessThanOrEqual(event.total);
          expect(event.loaded).toBeGreaterThanOrEqual(0);
          expect(event.total).toBeGreaterThan(0);
        }

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "per-file progress is monotonically non-decreasing",
      async () => {
        const { events, dispose } = await collectEvents((cb) => LlamaForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        // Group progress events by file and verify monotonicity within each file
        const progressByFile = {};
        for (const event of events.filter((e) => e.status === "progress")) {
          (progressByFile[event.file] ??= []).push(event.loaded);
        }
        for (const loadedValues of Object.values(progressByFile)) {
          for (let i = 1; i < loadedValues.length; i++) {
            expect(loadedValues[i]).toBeGreaterThanOrEqual(loadedValues[i - 1]);
          }
        }

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "progress_total files map is a deep copy (structuredClone)",
      async () => {
        const { events, dispose } = await collectEvents((cb) => LlamaForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        // Each progress_total event should have its own files object (not shared references)
        if (totalEvents.length >= 2) {
          expect(totalEvents[0].files).not.toBe(totalEvents[1].files);
        }

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });
});
