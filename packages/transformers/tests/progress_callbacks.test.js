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

describe("Progress Callbacks", () => {
  describe("Llama (decoder-only)", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    it(
      "pipeline('text-generation')",
      async () => {
        const { events, dispose } = await collectEvents((cb) => pipeline("text-generation", model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(0);

        // No double-wrapping: at most one progress_total per progress event
        const progressEvents = events.filter((e) => e.status === "progress");
        expect(totalEvents.length).toBeLessThanOrEqual(progressEvents.length);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "LlamaForCausalLM.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => LlamaForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(0);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "AutoModelForCausalLM.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => AutoModelForCausalLM.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(0);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  describe("Whisper (encoder-decoder)", () => {
    const model_id = "onnx-internal-testing/tiny-random-WhisperForConditionalGeneration";

    it(
      "pipeline('automatic-speech-recognition')",
      async () => {
        const { events, dispose } = await collectEvents((cb) => pipeline("automatic-speech-recognition", model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);

        // Encoder-decoder models should track multiple session files
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(1);

        // No double-wrapping
        const progressEvents = events.filter((e) => e.status === "progress");
        expect(totalEvents.length).toBeLessThanOrEqual(progressEvents.length);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );

    it(
      "WhisperForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => WhisperForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(1);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  describe("Gemma3 (image-text-to-text)", () => {
    const model_id = "onnx-internal-testing/tiny-random-Gemma3ForConditionalGeneration";

    it(
      "Gemma3ForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => Gemma3ForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(0);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  describe("Gemma3n (image-audio-text-to-text)", () => {
    const model_id = "onnx-internal-testing/tiny-random-Gemma3nForConditionalGeneration";

    it(
      "Gemma3nForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => Gemma3nForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(0);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });

  describe("VoxtralRealtime (audio-text-to-text)", () => {
    const model_id = "onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration";

    it(
      "VoxtralRealtimeForConditionalGeneration.from_pretrained()",
      async () => {
        const { events, dispose } = await collectEvents((cb) => VoxtralRealtimeForConditionalGeneration.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, progress_callback: cb }));

        const totalEvents = events.filter((e) => e.status === "progress_total");
        expectValidTotalEvents(totalEvents);
        expect(Object.keys(totalEvents.at(-1).files).length).toBeGreaterThan(0);

        await dispose();
      },
      MAX_MODEL_LOAD_TIME + MAX_MODEL_DISPOSE_TIME,
    );
  });
});
