// Walk a JS file with the TypeScript compiler and produce the structured
// entities (module header, classes with members, free functions, top-level
// constants, typedefs) that the IR layer groups into per-module docs.

import ts from "typescript";
import { parseJsDoc } from "./parse.mjs";

const FILE_LEVEL_TAGS = new Set(["module", "file", "typedef", "callback"]);

export function extractEntities(source, file) {
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const entities = { module: null, classes: [], functions: [], variables: [], typedefs: [] };

  // File-level blocks (`@module` / `@file` / standalone `@typedef` / `@callback`)
  // can't be discovered via node-attached JSDoc — TS attaches them to the next
  // statement. Scan the raw source and classify by tag.
  for (const m of source.matchAll(/\/\*\*[\s\S]*?\*\//g)) {
    if (isInsideLineComment(source, m.index)) continue;
    const parsed = parseJsDoc(m[0]);
    if (!parsed) continue;
    const tags = new Set(parsed.tags.map((t) => t.tag));
    if (tags.has("module") || tags.has("file")) entities.module ??= parsed;
    if (tags.has("typedef") || tags.has("callback")) entities.typedefs.push(parsed);
  }

  ts.forEachChild(sf, (node) => {
    if (ts.isClassDeclaration(node) && node.name) {
      entities.classes.push(buildClassEntity(node, sf));
    } else if (ts.isFunctionDeclaration(node) && node.name) {
      const parsed = parseNodeDoc(node);
      if (parsed && !hasFileLevelTag(parsed)) {
        entities.functions.push({ name: node.name.text, ...parsed });
      }
    } else if (ts.isVariableStatement(node)) {
      const parsed = parseNodeDoc(node);
      if (!parsed || hasFileLevelTag(parsed)) return;
      for (const decl of node.declarationList.declarations) {
        if (ts.isIdentifier(decl.name)) entities.variables.push({ name: decl.name.text, ...parsed });
      }
    }
  });

  return entities;
}

function buildClassEntity(node, sf) {
  const parsed = parseNodeDoc(node) ?? { description: "", tags: [] };
  const members = [];
  for (const m of node.members) {
    const member = buildMemberEntity(m, sf);
    if (member) members.push(member);
  }
  return { name: node.name.text, ...parsed, members };
}

function buildMemberEntity(node, sf) {
  const parsed = parseNodeDoc(node);
  if (!parsed) return null;

  if (ts.isConstructorDeclaration(node)) {
    return { kind: "method", name: "constructor", ...parsed };
  }
  if (ts.isMethodDeclaration(node) && node.name) {
    return { kind: "method", name: node.name.getText(sf), ...parsed };
  }
  if (ts.isPropertyDeclaration(node) && node.name) {
    return {
      kind: "field",
      name: node.name.getText(sf),
      initializer: node.initializer?.getText(sf) ?? null,
      ...parsed,
    };
  }
  if ((ts.isGetAccessor(node) || ts.isSetAccessor(node)) && node.name) {
    return {
      kind: ts.isGetAccessor(node) ? "getter" : "setter",
      name: node.name.getText(sf),
      ...parsed,
    };
  }
  return null;
}

// `ts.getJSDocCommentsAndTags` may return multiple blocks; the nearest/latest
// one is the canonical JSDoc for the node.
function parseNodeDoc(node) {
  const docs = ts.getJSDocCommentsAndTags(node).filter((d) => ts.isJSDoc(d));
  if (!docs.length) return null;
  return parseJsDoc(docs[docs.length - 1].getFullText());
}

function hasFileLevelTag(parsed) {
  return parsed.tags.some((t) => FILE_LEVEL_TAGS.has(t.tag));
}

function isInsideLineComment(source, offset) {
  const lineStart = source.lastIndexOf("\n", offset - 1) + 1;
  return /^\s*\/\//.test(source.slice(lineStart, offset));
}
