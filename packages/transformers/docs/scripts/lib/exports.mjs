// Resolve the symbol names the library exposes via `src/transformers.js`.
// Everything outside this set is treated as internal and excluded from docs.
//
// Public names are collected from:
//   - `export * from './path'` re-exports (walked recursively)
//   - `export { a, b as c }` named exports
//   - `export <decl> foo` direct declarations
//   - Properties of exported `Object.freeze({ X, Y, ... })` namespace objects,
//     so e.g. `random.Random` counts as public when `random` is exported.

import fs from "node:fs";
import path from "node:path";
import ts from "typescript";

export function collectPublicExports(entryFile) {
  const names = new Set();
  walk(path.resolve(entryFile), new Set(), names);
  return names;
}

function walk(file, visited, names) {
  if (visited.has(file) || !fs.existsSync(file)) return;
  visited.add(file);

  const source = fs.readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const dir = path.dirname(file);

  ts.forEachChild(sf, (node) => visitTopLevel(node, sf, dir, visited, names));
}

function visitTopLevel(node, sf, dir, visited, names) {
  // `export * from './path'`
  if (ts.isExportDeclaration(node) && !node.exportClause && node.moduleSpecifier) {
    const specifier = stripQuotes(node.moduleSpecifier.getText(sf));
    if (specifier.startsWith(".")) walk(path.resolve(dir, specifier), visited, names);
    return;
  }

  // `export { a, b as c }` or `export { a, b as c } from './path'`
  if (ts.isExportDeclaration(node) && node.exportClause && ts.isNamedExports(node.exportClause)) {
    const specifier = node.moduleSpecifier ? stripQuotes(node.moduleSpecifier.getText(sf)) : null;
    const importedNames = node.exportClause.elements.map((e) => (e.propertyName ?? e.name).text);
    for (const spec of node.exportClause.elements) names.add(spec.name.text);
    if (specifier?.startsWith(".")) {
      const target = path.resolve(dir, specifier);
      addNamespaceMembersFromFile(target, importedNames, names);
    }
    return;
  }

  if (!hasExportModifier(node)) return;

  if (ts.isClassDeclaration(node) || ts.isFunctionDeclaration(node)) {
    if (node.name) names.add(node.name.text);
    return;
  }

  if (ts.isVariableStatement(node)) {
    for (const decl of node.declarationList.declarations) {
      if (!ts.isIdentifier(decl.name)) continue;
      names.add(decl.name.text);
      addFrozenNamespaceMembers(decl.initializer, names);
    }
  }
}

// Look up the named exports in `file` and, for each one whose initializer is
// `Object.freeze({ ... })`, pull its shorthand-property names into `names`.
function addNamespaceMembersFromFile(file, wanted, names) {
  if (!fs.existsSync(file)) return;
  const source = fs.readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const wantedSet = new Set(wanted);
  ts.forEachChild(sf, (node) => {
    if (!ts.isVariableStatement(node) || !hasExportModifier(node)) return;
    for (const decl of node.declarationList.declarations) {
      if (ts.isIdentifier(decl.name) && wantedSet.has(decl.name.text)) {
        addFrozenNamespaceMembers(decl.initializer, names);
      }
    }
  });
}

// If the initializer is `Object.freeze({ ... })`, also treat each shorthand
// property name inside the object literal as publicly reachable via the
// exported namespace (e.g. `random.Random`).
function addFrozenNamespaceMembers(init, names) {
  const literal = unwrapObjectFreeze(init);
  if (!literal) return;
  for (const prop of literal.properties) {
    if (ts.isShorthandPropertyAssignment(prop)) names.add(prop.name.text);
  }
}

function unwrapObjectFreeze(init) {
  if (!init || !ts.isCallExpression(init)) return null;
  const callee = init.expression;
  const isFreeze =
    ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === "Object" && callee.name.text === "freeze";
  if (!isFreeze) return null;
  const arg = init.arguments[0];
  return arg && ts.isObjectLiteralExpression(arg) ? arg : null;
}

function hasExportModifier(node) {
  return !!node.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword);
}

function stripQuotes(s) {
  return s.replace(/^['"`]|['"`]$/g, "");
}
