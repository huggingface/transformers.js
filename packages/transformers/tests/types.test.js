import ts from "typescript";

describe("TypeScript compilation succeeds", () => {
  const DIR = "tests/types/";
  const FILES = ["pipelines.ts", "cache.ts"];
  for (const file of FILES) {
    it(`compiles ${file} without errors`, () => {
      const program = ts.createProgram([`${DIR}${file}`], {
        noEmit: true,
        skipLibCheck: true,
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ESNext,
      });

      const diagnostics = ts.getPreEmitDiagnostics(program);

      if (diagnostics.length > 0) {
        const formatHost = {
          getCanonicalFileName: (path) => path,
          getCurrentDirectory: ts.sys.getCurrentDirectory,
          getNewLine: () => ts.sys.newLine,
        };
        const message = ts.formatDiagnosticsWithColorAndContext(diagnostics, formatHost);
        throw new Error(message);
      }
    });
  }
});
