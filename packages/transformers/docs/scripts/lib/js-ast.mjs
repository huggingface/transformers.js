import ts from "typescript";

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
