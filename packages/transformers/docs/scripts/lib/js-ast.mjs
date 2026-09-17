import fs from "node:fs";
import ts from "typescript";

// Parse a JS source file into a TypeScript AST (with parent pointers).
export function parseJsFile(file) {
  return ts.createSourceFile(file, fs.readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
}

export function stripQuotes(text) {
  return text.replace(/^['"`]|['"`]$/g, "");
}

export function unwrapObjectFreeze(init) {
  if (!init || !ts.isCallExpression(init)) return null;
  const callee = init.expression;
  const isFreeze =
    ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === "Object" && callee.name.text === "freeze";
  if (!isFreeze) return null;
  const arg = init.arguments[0];
  return arg && ts.isObjectLiteralExpression(arg) ? arg : null;
}
