// Extract the task catalog (`SUPPORTED_TASKS` + `TASK_ALIASES`) from the
// library source. The skill renderer uses this to emit one task recipe per
// supported pipeline task, with its canonical pipeline class and default model.

import ts from "typescript";
import fs from "node:fs";

export function extractTaskCatalog(indexFile) {
  const source = fs.readFileSync(indexFile, "utf8");
  const sf = ts.createSourceFile(indexFile, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);

  const supportedTasks = new Map(); // task-id -> { pipelineClass, defaultModel, type }
  const aliases = new Map(); // alias -> task-id

  ts.forEachChild(sf, (node) => {
    if (!ts.isVariableStatement(node)) return;
    for (const decl of node.declarationList.declarations) {
      if (!ts.isIdentifier(decl.name)) continue;
      const literal = unwrapObjectFreeze(decl.initializer);
      if (!literal) continue;
      if (decl.name.text === "SUPPORTED_TASKS") parseSupportedTasks(literal, sf, supportedTasks);
      else if (decl.name.text === "TASK_ALIASES") parseAliases(literal, sf, aliases);
    }
  });

  return { supportedTasks, aliases };
}

// `Object.freeze({...})` -> the ObjectLiteralExpression argument, or null.
function unwrapObjectFreeze(init) {
  if (!init || !ts.isCallExpression(init)) return null;
  const arg = init.arguments[0];
  return arg && ts.isObjectLiteralExpression(arg) ? arg : null;
}

function parseSupportedTasks(literal, sf, out) {
  for (const prop of literal.properties) {
    if (!ts.isPropertyAssignment(prop)) continue;
    const task = stripQuotes(prop.name.getText(sf));
    if (!ts.isObjectLiteralExpression(prop.initializer)) continue;
    const entry = { pipelineClass: null, defaultModel: null, type: null };
    for (const p of prop.initializer.properties) {
      if (!ts.isPropertyAssignment(p)) continue;
      const key = p.name.getText(sf);
      if (key === "pipeline") entry.pipelineClass = p.initializer.getText(sf);
      else if (key === "type") entry.type = stripQuotes(p.initializer.getText(sf));
      else if (key === "default" && ts.isObjectLiteralExpression(p.initializer)) {
        for (const dp of p.initializer.properties) {
          if (ts.isPropertyAssignment(dp) && dp.name.getText(sf) === "model") {
            entry.defaultModel = stripQuotes(dp.initializer.getText(sf));
          }
        }
      }
    }
    out.set(task, entry);
  }
}

function parseAliases(literal, sf, out) {
  for (const prop of literal.properties) {
    if (!ts.isPropertyAssignment(prop)) continue;
    out.set(stripQuotes(prop.name.getText(sf)), stripQuotes(prop.initializer.getText(sf)));
  }
}

function stripQuotes(s) {
  return s.replace(/^['"`]|['"`]$/g, "");
}
