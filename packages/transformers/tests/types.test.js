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
  const FILES = ["pipelines.ts", "cache.ts"];
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
  /** @type {Record<string, { code: string, errors: Array<{ code: number, underline: string, messageIncludes: string }> }>} */
  const ERROR_CASES = {
    "past_key_values rejects string (chat input)": {
      code: `
        import { pipeline } from "../../types/transformers.js";
        const generator = await pipeline("text-generation", "model-id");
        const messages = [{ role: "user", content: "Hello!" }];
        await generator(messages, { past_key_values: "hello" });
      `,
      errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
    },
    "past_key_values rejects number (string input)": {
      code: `
        import { pipeline } from "../../types/transformers.js";
        const generator = await pipeline("text-generation", "model-id");
        await generator("hi", { past_key_values: 42 });
      `,
      errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
    },
    "past_key_values rejects boolean": {
      code: `
        import { pipeline } from "../../types/transformers.js";
        const generator = await pipeline("text-generation", "model-id");
        await generator("hi", { past_key_values: true });
      `,
      errors: [{ code: 2322, underline: "past_key_values", messageIncludes: "DynamicCache" }],
    },
    "past_key_values accepts DynamicCache": {
      code: `
        import { DynamicCache, pipeline } from "../../types/transformers.js";
        const generator = await pipeline("text-generation", "model-id");
        await generator("hi", { past_key_values: new DynamicCache() });
      `,
      errors: [],
    },
    "past_key_values accepts null": {
      code: `
        import { pipeline } from "../../types/transformers.js";
        const generator = await pipeline("text-generation", "model-id");
        await generator("hi", { past_key_values: null });
      `,
      errors: [],
    },
  };

  for (const [name, { code, errors }] of Object.entries(ERROR_CASES)) {
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
