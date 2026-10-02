import fs from "node:fs";
import ts from "typescript";

const inspectBundle = (name) => {
  const source = fs.readFileSync(new URL(`../dist/${name}`, import.meta.url), "utf8");
  const ast = ts.createSourceFile(name, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const imports = new Set();
  let hasImportMeta = false;
  const visit = (node) => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      imports.add(node.moduleSpecifier.text);
    } else if (ts.isCallExpression(node) && node.arguments.length && ts.isStringLiteral(node.arguments[0]) && (node.expression.kind === ts.SyntaxKind.ImportKeyword || (ts.isIdentifier(node.expression) && node.expression.text === "require"))) {
      imports.add(node.arguments[0].text);
    } else if (ts.isMetaProperty(node) && node.keywordToken === ts.SyntaxKind.ImportKeyword) {
      hasImportMeta = true;
    }
    ts.forEachChild(node, visit);
  };
  visit(ast);
  return { source, imports: [...imports], hasImportMeta };
};

describe("React Native bundle isolation", () => {
  it.each(["transformers.native.mjs", "transformers.native.min.mjs", "transformers.native.cjs", "transformers.native.min.cjs"])("keeps %s compatible with Metro/Hermes", (name) => {
    const { source, imports, hasImportMeta } = inspectBundle(name);
    expect(hasImportMeta).toBe(false);
    expect(imports).toEqual(expect.arrayContaining(["onnxruntime-react-native", "native-universal-fs"]));
    expect(imports.filter((specifier) => /^(node:|onnxruntime-(node|web)(\/|$)|sharp$|fs$|path$|url$|stream(\/|$))/.test(specifier))).toEqual([]);
    // The upstream Node wrapper invokes createRequire and process.env at module
    // evaluation time, so it must be excluded rather than merely ignoring ORT.
    expect(source).not.toContain("ORT_DISABLE_TELEMETRY");
  });

  it.each(["transformers.js", "transformers.web.js", "transformers.node.mjs", "transformers.node.cjs"])("does not expose React Native dependencies in %s", (name) => {
    const { imports } = inspectBundle(name);
    expect(imports.filter((specifier) => /^(onnxruntime-react-native|native-universal-fs|react-native)(\/|$)/.test(specifier))).toEqual([]);
  });
});
