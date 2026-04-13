import ts from "typescript";

const TS_OPTIONS = {
  noEmit: true,
  skipLibCheck: true,
  module: ts.ModuleKind.ESNext,
  target: ts.ScriptTarget.ESNext,
};

function getDiagnostics(file) {
  const program = ts.createProgram([file], TS_OPTIONS);
  return ts.getPreEmitDiagnostics(program);
}

function formatDiagnostics(diagnostics) {
  const formatHost = {
    getCanonicalFileName: (path) => path,
    getCurrentDirectory: ts.sys.getCurrentDirectory,
    getNewLine: () => ts.sys.newLine,
  };
  return ts.formatDiagnosticsWithColorAndContext(diagnostics, formatHost);
}

/**
 * Compile an inline TypeScript source string and return diagnostics.
 * Uses a virtual file path under tests/types/ so relative imports resolve correctly.
 */
function getDiagnosticsFromSource(source) {
  const virtualPath = "tests/types/__virtual.ts";
  const defaultHost = ts.createCompilerHost(TS_OPTIONS);
  const host = {
    ...defaultHost,
    fileExists: (f) => f === virtualPath || defaultHost.fileExists(f),
    readFile: (f) => (f === virtualPath ? source : defaultHost.readFile(f)),
    getSourceFile: (f, lang) => (f === virtualPath ? ts.createSourceFile(f, source, lang, true) : defaultHost.getSourceFile(f, lang)),
  };
  const program = ts.createProgram([virtualPath], TS_OPTIONS, host);
  return [...ts.getPreEmitDiagnostics(program)];
}

describe("TypeScript compilation succeeds", () => {
  const DIR = "tests/types/";
  const FILES = ["pipelines.ts", "cache.ts", "tokenizers.ts"];
  for (const file of FILES) {
    it(`compiles ${file} without errors`, () => {
      const diagnostics = getDiagnostics(`${DIR}${file}`);
      if (diagnostics.length > 0) {
        throw new Error(formatDiagnostics(diagnostics));
      }
    });
  }
});

describe("TypeScript expected errors", () => {
  /**
   * Helper: generates inline source for a pipeline error test.
   * @param {string} task Pipeline task name
   * @param {string} callArgs Arguments to pass to the pipeline call (e.g., `'"hi", { top_k: "bad" }')`)
   */
  const src = (task, callArgs) => `
    import { pipeline } from "../../types/transformers.js";
    const p = await pipeline("${task}", "model-id");
    await p(${callArgs});
  `;

  /** @param {string} task @param {string} callArgs */
  const srcWithDynamicCache = (task, callArgs) => `
    import { DynamicCache, pipeline } from "../../types/transformers.js";
    const p = await pipeline("${task}", "model-id");
    await p(${callArgs});
  `;

  /**
   * Pipeline error test cases.
   *
   * Each pipeline task maps to an array of test cases. Each test case has:
   * - `name`: test description
   * - `code`: inline TypeScript source (use `src()` helper for convenience)
   * - `errors`: expected diagnostics (empty = should compile without errors)
   *
   * @type {Record<string, Array<{ name: string, code: string, errors: Array<{ code: number, underline: string, messageIncludes: string }> }>>}
   */
  const PIPELINE_CASES = {
    "text-generation": [
      {
        name: "past_key_values rejects string (chat input)",
        code: src("text-generation", `[{role:"user",content:"hi"}], { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
      {
        name: "past_key_values rejects string (string input)",
        code: src("text-generation", `"hi", { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
      {
        name: "past_key_values rejects number",
        code: src("text-generation", `"hi", { past_key_values: 42 }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
      {
        name: "past_key_values rejects boolean",
        code: src("text-generation", `"hi", { past_key_values: true }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
      {
        name: "past_key_values accepts DynamicCache",
        code: srcWithDynamicCache("text-generation", `"hi", { past_key_values: new DynamicCache() }`),
        errors: [],
      },
      {
        name: "past_key_values accepts null",
        code: src("text-generation", `"hi", { past_key_values: null }`),
        errors: [],
      },
    ],

    "text2text-generation": [
      {
        name: "past_key_values rejects string",
        code: src("text2text-generation", `"translate: hello", { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
    ],

    translation: [
      {
        name: "past_key_values rejects string",
        code: src("translation", `"Bonjour", { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
    ],

    summarization: [
      {
        name: "past_key_values rejects string",
        code: src("summarization", `"Long article text...", { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
    ],

    "image-to-text": [
      {
        name: "past_key_values rejects string",
        code: src("image-to-text", `"http://img.png", { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
    ],

    "automatic-speech-recognition": [
      {
        name: "past_key_values rejects string",
        code: src("automatic-speech-recognition", `new Float32Array(16000), { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
    ],

    "document-question-answering": [
      {
        name: "past_key_values rejects string",
        code: src("document-question-answering", `"http://img.png", "What is this?", { past_key_values: "bad" }`),
        errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
      },
    ],

    "audio-classification": [
      {
        name: "top_k rejects string",
        code: src("audio-classification", `new Float32Array(16000), { top_k: "bad" }`),
        errors: [{ code: 2322, underline: "top_k", messageIncludes: "number" }],
      },
    ],

    "fill-mask": [
      {
        name: "top_k rejects string",
        code: src("fill-mask", `"The [MASK] sat on the mat.", { top_k: "bad" }`),
        errors: [{ code: 2322, underline: "top_k", messageIncludes: "number" }],
      },
    ],

    "image-classification": [
      {
        name: "top_k rejects string",
        code: src("image-classification", `"http://img.png", { top_k: "bad" }`),
        errors: [{ code: 2322, underline: "top_k", messageIncludes: "number" }],
      },
    ],

    "object-detection": [
      {
        name: "threshold rejects string",
        code: src("object-detection", `"http://img.png", { threshold: "bad" }`),
        errors: [{ code: 2322, underline: "threshold", messageIncludes: "number" }],
      },
    ],

    "token-classification": [
      {
        name: "ignore_labels rejects string",
        code: src("token-classification", `"Hello world", { ignore_labels: "bad" }`),
        errors: [{ code: 2322, underline: "ignore_labels", messageIncludes: "string[]" }],
      },
    ],

    "text-classification": [
      {
        name: "top_k rejects string",
        code: src("text-classification", `"I love this!", { top_k: "bad" }`),
        errors: [{ code: 2322, underline: "top_k", messageIncludes: "number" }],
      },
    ],

    "question-answering": [
      {
        name: "top_k rejects string",
        code: src("question-answering", `"Who?", "Context.", { top_k: "bad" }`),
        errors: [{ code: 2322, underline: "top_k", messageIncludes: "number" }],
      },
    ],

    "feature-extraction": [
      {
        name: "pooling rejects number",
        code: src("feature-extraction", `"text", { pooling: 123 }`),
        errors: [{ code: 2322, underline: "pooling", messageIncludes: '"none" | "mean"' }],
      },
    ],

    "image-segmentation": [
      {
        name: "threshold rejects string",
        code: src("image-segmentation", `"http://img.png", { threshold: "bad" }`),
        errors: [{ code: 2322, underline: "threshold", messageIncludes: "number" }],
      },
    ],

    "image-feature-extraction": [
      {
        name: "pool rejects string",
        code: src("image-feature-extraction", `"http://img.png", { pool: "bad" }`),
        errors: [{ code: 2322, underline: "pool", messageIncludes: "boolean" }],
      },
    ],

    "text-to-audio": [
      {
        name: "speaker_embeddings rejects number",
        code: src("text-to-audio", `"Hello", { speaker_embeddings: 123 }`),
        errors: [{ code: 2322, underline: "speaker_embeddings", messageIncludes: "Tensor" }],
      },
    ],

    "zero-shot-classification": [
      {
        name: "multi_label rejects string",
        code: src("zero-shot-classification", `"text", ["a", "b"], { multi_label: "bad" }`),
        errors: [{ code: 2322, underline: "multi_label", messageIncludes: "boolean" }],
      },
    ],

    "zero-shot-audio-classification": [
      {
        name: "hypothesis_template rejects number",
        code: src("zero-shot-audio-classification", `new Float32Array(16000), ["a", "b"], { hypothesis_template: 123 }`),
        errors: [{ code: 2322, underline: "hypothesis_template", messageIncludes: "string" }],
      },
    ],

    "zero-shot-image-classification": [
      {
        name: "hypothesis_template rejects number",
        code: src("zero-shot-image-classification", `"http://img.png", ["a", "b"], { hypothesis_template: 123 }`),
        errors: [{ code: 2322, underline: "hypothesis_template", messageIncludes: "string" }],
      },
    ],

    "zero-shot-object-detection": [
      {
        name: "threshold rejects string",
        code: src("zero-shot-object-detection", `"http://img.png", ["a", "b"], { threshold: "bad" }`),
        errors: [{ code: 2322, underline: "threshold", messageIncludes: "number" }],
      },
    ],

    "depth-estimation": [
      {
        name: "rejects unexpected options argument",
        code: src("depth-estimation", `"http://img.png", { bad: true }`),
        errors: [{ code: 2554, underline: "{ bad: true }", messageIncludes: "Expected 1 arguments" }],
      },
    ],

    "image-to-image": [
      {
        name: "rejects unexpected options argument",
        code: src("image-to-image", `"http://img.png", { bad: true }`),
        errors: [{ code: 2554, underline: "{ bad: true }", messageIncludes: "Expected 1 arguments" }],
      },
    ],
  };

  describe("pipeline errors", () => {
    for (const [task, cases] of Object.entries(PIPELINE_CASES)) {
      describe(task, () => {
        for (const { name, code, errors } of cases) {
          it(name, () => {
            const diagnostics = getDiagnosticsFromSource(code);

            expect(diagnostics).toHaveLength(errors.length);

            for (let i = 0; i < errors.length; i++) {
              const diag = diagnostics[i];
              const expected = errors[i];

              expect(diag.code).toBe(expected.code);

              const underlined = diag.file?.text?.slice(diag.start, diag.start + diag.length);
              expect(underlined).toBe(expected.underline);

              const message = ts.flattenDiagnosticMessageText(diag.messageText, "\n");
              expect(message).toContain(expected.messageIncludes);
            }
          });
        }
      });
    }
  });
});
