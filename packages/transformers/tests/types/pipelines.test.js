import ts from "typescript";

describe("Testing pipeline types", () => {
  it("TypeScript compilation succeeds", () => {
    const program = ts.createProgram(["tests/types/pipelines.ts"], {
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
});
