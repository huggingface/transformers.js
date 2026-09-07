// Render a module's IR into readable markdown. Principles:
// - Prefer typedef names over their expanded TS structure
// - Never emit raw HTML tables or `&lt;code&gt;` escaping
// - Filter to the library's public export surface

import path from "node:path";

import { apiMemberAnchor, apiSymbolAnchor } from "./api-links.mjs";
import { matchingBracket, splitConditional, splitTopLevel } from "./scan.mjs";
import { DOC_REFERENCE, exampleLines, firstSentence, paramSignature, stripImportPrefixes, transformOutsideFences } from "./text.mjs";
import { parseCallableReference, parseUtilityType, TS_UTILITY_NAMES } from "./type-refs.mjs";

// Pages with at least this many top-level items get an "On this page" TOC.
const TOC_THRESHOLD = 6;

// True when the module has reader-visible content: a public class/function/constant, or a renderable
// typedef/callback. Lets `generateApiDocs` skip internal-only modules without rendering them.
export function hasRenderableContent(mod, publicNames = null) {
  if (filterPublic(mod.classes, publicNames).length) return true;
  if (filterPublic(mod.functions, publicNames).length) return true;
  if (filterPublic(mod.constants, publicNames).length) return true;
  if (mod.callbacks.length) return true;
  return mod.typedefs.some((td) => !isInternalTypedef(td));
}

// The indexes depend only on `(ir, publicNames)`: build once and pass to `renderModule` via
// `opts.linkIndexes` when rendering many pages.
export function buildLinkIndexes(ir, publicNames = null) {
  return {
    renderedNames: buildRenderedNameIndex(ir, publicNames),
    callableLinks: buildCallableLinkIndex(ir, publicNames),
  };
}

export function renderModule(mod, ir, opts) {
  const publicNames = opts.publicNames ?? null;
  const { renderedNames, callableLinks } = opts.linkIndexes;
  const ctx = {
    typedefIndex: ir.typedefIndex,
    moduleByName: new Map(ir.modules.map((m) => [m.name, m])),
    moduleName: mod.name,
    // Unfiltered: `renderTypedef` needs the module's `_`-prefixed classes, which the public
    // export surface omits.
    moduleClasses: mod.classes,
    renderedNames,
    callableLinks,
  };

  const classes = filterPublic(mod.classes, publicNames);
  const functions = filterPublic(mod.functions, publicNames);
  const constants = dedupeByName(filterPublic(mod.constants, publicNames));

  const out = [];
  out.push(`# ${mod.name}`, "");
  if (mod.description) out.push(mod.description.trim(), "");
  for (const ex of mod.examples) out.push(...exampleLines(ex));

  out.push(...renderTOC({ classes, functions, constants }, ctx));

  if (classes.length) {
    out.push("## Classes", "");
    for (const cls of classes) out.push(...renderClass(cls, ctx));
  }
  if (functions.length) {
    out.push("## Functions", "");
    for (const fn of functions) out.push(...renderFunction(fn, ctx, 3));
  }
  if (constants.length) {
    out.push("## Constants", "");
    for (const c of constants) out.push(...renderConstant(c, ctx));
  }
  if (mod.typedefs.length) {
    const rendered = mod.typedefs.flatMap((td) => renderTypedef(td, ctx));
    if (rendered.length) out.push("## Type Definitions", "", ...rendered);
  }
  if (mod.callbacks.length) {
    out.push("## Callbacks", "");
    for (const cb of mod.callbacks) out.push(...renderFunction(cb, ctx, 3, null, { nameOnlyHeading: true }));
  }

  // Link expansion and blank-run collapsing must not touch example code, so fenced blocks are
  // masked; trailing-whitespace cleanup is safe everywhere.
  return transformOutsideFences(out.join("\n").replace(/[ \t]+$/gm, ""), (text) => expandInlineLinks(text, ctx).replace(/\n{3,}/g, "\n\n")).trimEnd() + "\n";
}

// Typedefs and callbacks are a long tail that would dominate the TOC, so only classes, functions
// and constants are listed; readers reach the rest via the section heading.
function renderTOC({ classes, functions, constants }, ctx) {
  const total = classes.length + functions.length + constants.length;
  if (total < TOC_THRESHOLD) return [];

  const groups = [
    ["Classes", classes, (c) => apiSymbolAnchor(ctx.moduleName, c.name)],
    ["Functions", functions, (f) => apiSymbolAnchor(ctx.moduleName, f.name)],
    ["Constants", constants, (c) => apiSymbolAnchor(ctx.moduleName, c.name)],
  ];

  const lines = ["## On this page", ""];
  for (const [title, items, anchorOf] of groups) {
    if (!items.length) continue;
    const links = items.map((it) => `[\`${it.name}\`](#${anchorOf(it)})`).join(" · ");
    lines.push(`**${title}** — ${links}`, "");
  }
  return lines;
}

function buildRenderedNameIndex(ir, publicNames) {
  const names = new Set();
  const ctx = { typedefIndex: ir.typedefIndex, renderedNames: names };
  for (const mod of ir.modules) {
    for (const cls of filterPublic(mod.classes, publicNames)) names.add(cls.name);
    for (const cb of mod.callbacks) names.add(cb.name);
    for (const td of mod.typedefs) {
      if (shouldRenderTypedef(td, { ...ctx, moduleName: mod.name })) names.add(td.name);
    }
  }
  return names;
}

function buildCallableLinkIndex(ir, publicNames) {
  const links = new Map();
  for (const mod of ir.modules) {
    for (const fn of filterPublic(mod.functions, publicNames)) {
      links.set(fn.name, { moduleName: mod.name, anchor: apiSymbolAnchor(mod.name, fn.name) });
    }
    for (const cls of filterPublic(mod.classes, publicNames)) {
      for (const m of cls.members) {
        if (m.kind === "method" && shouldRenderMethod(m)) {
          links.set(`${cls.name}.${m.name}`, { moduleName: mod.name, anchor: apiMemberAnchor(mod.name, cls.name, m.name) });
        }
      }
    }
  }
  return links;
}

function filterPublic(items, publicNames) {
  return publicNames ? items.filter((it) => publicNames.has(it.name)) : items;
}

function dedupeByName(items) {
  const seen = new Set();
  return items.filter((it) => (seen.has(it.name) ? false : seen.add(it.name)));
}

// reST-style ``[`TypeName`]`` references are left alone; `expandInlineLinks` resolves them once the
// page-level symbol index exists. `@see Symbol` becomes inline markup so it reads as prose.
function cleanDescription(text) {
  return text
    .trim()
    .replace(/@see\s+(?=\{@link)/g, "")
    .replace(/@see\s+`?([A-Za-z_$][\w$.]*)`?/g, "`$1`");
}

// Line-leading `@see` tags land in `entity.see`; any `{@link ...}` inside them is expanded later by
// `expandInlineLinks`.
function seeAlso(entity) {
  return entity.see?.length ? [`**See also:** ${entity.see.join(", ")}`, ""] : [];
}

// `{@link url}` / `{@link url Text}` / `{@link Symbol}` -> markdown.
function expandInlineLinks(text, ctx) {
  return text
    .replace(/\{@link\s+([^}\s]+)(?:\s+([^}]+))?\}/g, (_, target, label) => {
      const displayed = (label ?? target).trim();
      return /^https?:\/\//.test(target) ? `[${displayed}](${target})` : (linkReference(target, displayed, ctx) ?? `\`${displayed}\``);
    })
    .replace(DOC_REFERENCE, (_, ref) => {
      // `pipeline()` reads as a call but resolves as the symbol `pipeline`.
      const symbol = ref.replace(/\(\)$/, "");
      return linkReference(symbol, ref, ctx) ?? `\`${ref}\``;
    });
}

// ---------- classes, functions, callbacks ----------

function renderClass(cls, ctx) {
  const lines = [`<a id="${apiSymbolAnchor(ctx.moduleName, cls.name)}"></a>`, "", `### ${cls.name}`, ""];
  if (cls.description) lines.push(cleanDescription(cls.description), "");
  lines.push(...seeAlso(cls));
  for (const ex of cls.examples) lines.push(...exampleLines(ex));
  if (cls.callable) {
    lines.push(...renderFunction({ ...cls.callable, displayName: cls.name, anchorName: `${cls.name}.call` }, ctx, 4));
  }

  for (const m of cls.members) {
    if (m.kind === "method") {
      if (!shouldRenderMethod(m)) continue;
      lines.push(...renderFunction(m, ctx, 4, cls.name));
    } else {
      lines.push(...renderField(m, ctx, cls.name));
    }
  }
  return lines;
}

function renderField(f, ctx, parent) {
  if (f.name.startsWith("_")) return [];
  // Undocumented, untyped placeholders carry nothing; a deprecation flag still keeps a field
  // visible as a warning.
  if (!f.type && !f.description && !f.deprecated && f.defaultValue == null) return [];

  const type = f.type ? ` : ${renderType(f.type, ctx)}` : "";
  const lines = [`#### \`${parent}.${f.name}\`${type}`, ""];
  if (f.deprecated) lines.push("> **Deprecated**", "");
  if (f.description) lines.push(cleanDescription(f.description), "");
  // Multiline initializers can't sit inside a single-backtick span — omit them.
  if (f.defaultValue != null && f.defaultValue !== "" && !f.defaultValue.includes("\n")) {
    lines.push(`**Default:** \`${f.defaultValue}\``, "");
  }
  return lines;
}

// `_`-prefixed methods are internal / subclass-only hooks by convention, so they stay out of the
// user-facing API.
export function shouldRenderMethod(m) {
  if (m.name.startsWith("_")) return false;
  return m.description || m.params?.length || m.returns?.description || m.returns?.type || m.examples?.length || m.throws?.length;
}

// `opts.nameOnlyHeading` renders callback typedefs: they are types rather than invocable
// functions, so the heading is the bare name and not a `name(params)` signature.
function renderFunction(fn, ctx, depth, parent = null, opts = {}) {
  const anchor = parent ? apiMemberAnchor(ctx.moduleName, parent, fn.anchorName ?? fn.name) : apiSymbolAnchor(ctx.moduleName, fn.anchorName ?? fn.name);
  const heading = opts.nameOnlyHeading ? fn.name : signature(fn, parent);
  const lines = [`<a id="${anchor}"></a>`, "", `${"#".repeat(depth)} ${heading}`, ""];
  // Without the constraint map, a `T` in the parameter/return types would render as `any`.
  const fnCtx = withTemplates(ctx, fn);

  if (fn.deprecated) lines.push("> **Deprecated**", "");
  if (fn.description) lines.push(cleanDescription(fn.description), "");
  lines.push(...seeAlso(fn));
  if (fn.params?.length) {
    lines.push("**Parameters**", "", ...renderParamList(fn.params, fnCtx), "");
  }
  if (fn.returns?.type || fn.returns?.description) {
    const type = fn.returns.type ? renderType(fn.returns.type, fnCtx) : "";
    const desc = fn.returns.description ? (type ? ` — ${fn.returns.description.trim()}` : fn.returns.description.trim()) : "";
    lines.push(`**Returns:** ${type}${desc}`.trim(), "");
  }
  if (fn.throws?.length) {
    lines.push("**Throws**", "");
    for (const t of fn.throws) {
      const type = t.type ? `\`${prettifyTypeString(t.type)}\`` : "";
      const sep = type && t.description ? " — " : "";
      lines.push(`- ${type}${sep}${t.description?.trim() ?? ""}`);
    }
    lines.push("");
  }
  for (const ex of fn.examples ?? []) lines.push(...exampleLines(ex));
  return lines;
}

function signature(fn, parent) {
  const owner = parent ? `${parent}.` : "";
  return `\`${owner}${fn.displayName ?? fn.name}(${paramSignature(fn.params)})\``;
}

// ---------- parameter lists ----------

// Nested params (`foo.bar`) fold under their parent as sub-bullets.
function renderParamList(params, ctx) {
  const roots = [];
  const byName = new Map();
  for (const p of params) {
    if (!p.name) continue;
    const node = { ...p, children: [] };
    const dot = p.name.lastIndexOf(".");
    const parent = dot >= 0 ? byName.get(p.name.slice(0, dot)) : null;
    if (parent) parent.children.push(node);
    else roots.push(node);
    byName.set(p.name, node);
  }
  return roots.flatMap((p) => renderParamNode(p, ctx, 0));
}

function renderParamNode(p, ctx, indent) {
  const pad = "  ".repeat(indent);
  const contPad = "  ".repeat(indent + 1);
  const type = p.type ? ` (${renderType(p.type, ctx)})` : "";
  const opt = p.optional ? " _optional_" : "";
  const def = p.defaultValue != null ? ` — defaults to \`${p.defaultValue}\`` : "";
  // Parameters typed as a named typedef and given no inline description borrow that typedef's
  // first sentence, rather than rendering as a bare `- name (Type) optional`.
  const fallback = p.description ? null : typedefSummary(p.type, ctx);
  const descText = p.description?.trim() || fallback;
  const desc = descText ? ` — ${indentContinuations(descText, contPad)}` : "";
  const line = `${pad}- \`${simpleName(p.name)}\`${type}${opt}${def}${desc}`;
  return [line, ...p.children.flatMap((c) => renderParamNode(c, ctx, indent + 1))];
}

// First sentence of the named typedef's description. Skips wrappers (unions, arrays, generics) to
// avoid misleading borrows.
function typedefSummary(rawType, ctx) {
  if (!rawType || !ctx.typedefIndex) return null;
  const pretty = prettifyTypeString(rawType);
  if (!SIMPLE_NAME.test(pretty)) return null;
  const moduleName = ctx.typedefIndex.get(pretty);
  if (!moduleName) return null;
  const mod = ctx.moduleByName?.get(moduleName);
  const def = mod?.typedefs.find((t) => t.name === pretty) ?? mod?.callbacks.find((c) => c.name === pretty);
  if (!def?.description) return null;
  return firstSentence(def.description);
}

// Indent continuation lines so multi-line descriptions render inside their parent list item.
function indentContinuations(text, pad) {
  const lines = text.split("\n");
  if (lines.length === 1) return text;
  return (
    lines[0] +
    "\n" +
    lines
      .slice(1)
      .map((l) => pad + l.replace(/^\s+/, ""))
      .join("\n")
  );
}

function simpleName(name) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot + 1) : name;
}

// ---------- constants & typedefs ----------

function renderConstant(c, ctx) {
  const type = c.type ? ` : ${renderType(c.type, ctx)}` : "";
  const lines = [`<a id="${apiSymbolAnchor(ctx.moduleName, c.name)}"></a>`, "", `### \`${c.name}\`${type}`, ""];
  if (c.description) lines.push(cleanDescription(c.description), "");
  lines.push(...seeAlso(c));
  for (const ex of c.examples) lines.push(...exampleLines(ex));
  return lines;
}

// Skip typedefs that only thread generic parameters through the type system (`@typedef {T} Name`,
// `_`-prefixed names) or that are opaque placeholders with nothing to show.
function isInternalTypedef(td) {
  if (td.name.startsWith("_")) return true;
  const type = (td.type ?? "").trim();
  // `@typedef {object} Foo` with no body — nothing to show.
  if ((type === "object" || type === "Object") && !td.description && !td.properties?.length) return true;
  // `@typedef {GenericParam} Foo` — bare alias with no description or body.
  if (/^[A-Za-z_$][\w$.]*$/.test(type) && !td.description && !td.properties?.length) return true;
  return false;
}

// Generic type parameter names (T, K, V, TItem, ...). Rendered as `any` so the reader doesn't
// chase an undefined symbol.
function isGenericParamName(name) {
  return /^T[A-Z][A-Za-z]*$/.test(name) || /^[TKV]$/.test(name);
}

// Constraint map for `@template {Constraint} T`; an unconstrained `T` renders as `any`.
function withTemplates(ctx, entity) {
  const templateMap = new Map();
  for (const t of entity?.templates ?? []) if (t.name && t.type) templateMap.set(t.name, t.type);
  return templateMap.size ? { ...ctx, templates: templateMap } : ctx;
}

function renderTypedef(td, ctx) {
  if (!shouldRenderTypedef(td, ctx)) return [];

  const tdCtx = withTemplates(ctx, td);
  const { displayed, typeIsShowable } = typedefRenderInfo(td, ctx);

  const lines = [`<a id="${apiSymbolAnchor(ctx.moduleName, td.name)}"></a>`, "", `### ${td.name}`, ""];
  if (td.description) lines.push(cleanDescription(td.description), "");
  if (typeIsShowable) {
    lines.push(`_Type:_ ${displayed}`, "");
  }
  if (td.properties?.length) {
    lines.push("**Properties**", "", ...renderParamList(td.properties, tdCtx), "");
  }
  lines.push(...renderBackingClassMembers(td, ctx));
  return lines;
}

// `@typedef {Record<string, Tensor> & _DynamicCache} DynamicCache` means "a plain object that also
// has these methods": the members live on a `_`-prefixed class that is never documented on its own.
// Find that class so its members render under the typedef.
function backingClass(td, ctx) {
  if (!td.type || !ctx.moduleClasses?.length) return null;
  const parts = splitTopLevel(prettifyTypeString(td.type), "&").map((p) => p.trim());
  if (parts.length < 2) return null;
  for (const part of parts) {
    if (!/^_[A-Za-z]/.test(part)) continue;
    const cls = ctx.moduleClasses.find((c) => c.name === part);
    if (cls) return cls;
  }
  return null;
}

// Anchored to the typedef's name (`module_x.DynamicCache.update`) so links point at the name
// readers use. The `_Type:_` line above already shows the non-private half of the intersection.
function renderBackingClassMembers(td, ctx) {
  const cls = backingClass(td, ctx);
  if (!cls) return [];
  return cls.members.flatMap((m) => (m.kind === "method" && shouldRenderMethod(m) ? renderFunction(m, ctx, 4, td.name) : []));
}

function shouldRenderTypedef(td, ctx) {
  if (isInternalTypedef(td)) return false;
  const { typeIsShowable } = typedefRenderInfo(td, ctx);
  return !!(td.description || td.properties?.length || typeIsShowable);
}

function typedefRenderInfo(td, ctx) {
  const displayed = td.type ? renderType(td.type, { ...withTemplates(ctx, td), selfName: td.name }) : "";
  const isSelfReference = displayed === `\`${td.name}\``;
  const isGenericPassthrough = /^`[A-Z][A-Za-z]?`$/.test(displayed);
  // Collapsed fallbacks carry no information: `_Type:_ `object`` is noise.
  const isOpaque = ["`object`", "`Object`", "`unknown`", "`any`"].includes(displayed);
  // Unions and intersections are shown even when long: every variant is a clickable named type. An
  // object literal has no clickable parts, so it must still fit on a line. Typedefs with a
  // **Properties** list show that instead.
  const isUnionOrIntersection = / \| | & /.test(displayed);
  const isObjectLiteral = displayed.startsWith("`{");
  const fitsInline = displayed.length < 120;
  // `_`-prefixed names never get a page of their own, so a type built from one points nowhere.
  const referencesInternal = /`_[A-Za-z]/.test(displayed);
  const typeIsShowable =
    displayed &&
    !td.properties?.length &&
    (fitsInline || (isUnionOrIntersection && !isObjectLiteral)) &&
    !isSelfReference &&
    !isGenericPassthrough &&
    !isOpaque &&
    !referencesInternal;

  return { displayed, typeIsShowable };
}

// ---------- type rendering ----------

const GENERIC_WRAPPERS = /^(Promise|Array|Record|Map|Set|Iterable|AsyncIterable|Partial|Readonly)<(.+)>$/;
const NAMED_GENERIC = /^([A-Za-z_$][\w$.]*)<(.+)>$/;
const SIMPLE_NAME = new RegExp(`^[A-Za-z_$][\\w$.]*$`);
const INDEXED_NAME = /^([A-Za-z_$][\w$.]*)\[[^\]]+\]$/;
const TUPLE = /^\[(.*)\]$/;

// Turn a raw JSDoc type string into readable markdown. Prefers author-written names over expanded
// TS structures; gnarly types become `object`.
function renderType(raw, ctx, opts = {}) {
  const pretty = prettifyTypeString(raw, ctx?.selfName);
  if (opts.noLink) return `\`${pretty}\``;

  const utility = parseUtilityType(pretty);
  if (utility) return renderUtilityType(utility, ctx);

  // Unions split first. `_`-prefixed variants (internal types from intersections) are dropped so
  // they don't leak into the public docs.
  const unionParts = splitTopLevel(pretty, "|")
    .map((p) => p.trim())
    .filter((p) => !/^_[A-Za-z]/.test(p));
  if (unionParts.length > 1) {
    // `T | T[]` stays two parts: a collapsed `T[]?` reads as "optional array", which the union
    // does not mean.
    return unionParts.map((p) => renderType(p, ctx)).join(" | ");
  }
  if (unionParts.length === 1 && unionParts[0] !== pretty.trim()) return renderType(unionParts[0], ctx);

  // Same for intersections (`A & B`).
  const intersectParts = splitTopLevel(pretty, "&")
    .map((p) => p.trim())
    .filter((p) => !/^_[A-Za-z]/.test(p));
  if (intersectParts.length > 1) return intersectParts.map((p) => renderType(p, ctx)).join(" & ");
  if (intersectParts.length === 1 && intersectParts[0] !== pretty.trim()) return renderType(intersectParts[0], ctx);

  if (pretty.endsWith("[]")) {
    return renderArrayType(pretty.slice(0, -2), ctx);
  }

  const tuple = pretty.match(TUPLE);
  if (tuple) return renderTupleType(tuple[1], ctx);

  if (SIMPLE_NAME.test(pretty)) {
    // `@template {Constraint} T` — render `T` as its constraint when we know it.
    if (ctx.templates?.has(pretty)) return renderType(ctx.templates.get(pretty), { ...ctx, templates: undefined });
    if (isGenericParamName(pretty)) return "`any`";
    return linkIfKnown(pretty, ctx) ?? `\`${pretty}\``;
  }

  const indexed = pretty.match(INDEXED_NAME);
  if (indexed) {
    const linked = linkIfKnown(indexed[1], ctx);
    if (linked) return linked;
  }

  const wrapper = pretty.match(GENERIC_WRAPPERS);
  if (wrapper) {
    const innerParts = splitTopLevel(wrapper[2], ",");
    const rendered = innerParts.map((p) => renderType(p.trim(), ctx)).join(", ");
    return `\`${wrapper[1]}\`<${rendered}>`;
  }

  const namedGeneric = pretty.match(NAMED_GENERIC);
  if (namedGeneric && ctx.typedefIndex?.has(namedGeneric[1])) {
    const innerParts = splitTopLevel(namedGeneric[2], ",");
    const rendered = innerParts.map((p) => renderGenericArgument(p.trim(), ctx)).join(", ");
    const outer = linkIfKnown(namedGeneric[1], ctx) ?? `\`${namedGeneric[1]}\``;
    return `${outer}<${rendered}>`;
  }

  return `\`${pretty}\``;
}

function renderGenericArgument(raw, ctx) {
  if (SIMPLE_NAME.test(raw) && isGenericParamName(raw) && !ctx.templates?.has(raw)) {
    return `\`${raw}\``;
  }
  return renderType(raw, ctx);
}

function renderArrayType(innerRaw, ctx) {
  const rendered = renderType(innerRaw, ctx);
  const code = rendered.match(/^`([^`]+)`$/);
  return code ? `\`${code[1]}[]\`` : `${rendered}[]`;
}

// One code span with plain member names: ``[`number`, `number`]`` is indistinguishable from a
// broken markdown link.
function renderTupleType(innerRaw, ctx) {
  const parts = splitTopLevel(innerRaw, ",");
  if (parts.length === 1 && !parts[0].trim()) return "`[]`";
  const rendered = parts.map((p) => stripMarkup(renderType(p.trim(), ctx)));
  return `\`[${rendered.join(", ")}]\``;
}

// Reduce rendered markdown back to bare type text so it can sit inside an enclosing code span.
function stripMarkup(rendered) {
  return rendered.replace(/\[`([^`]+)`\]\([^)]*\)/g, "$1").replace(/`/g, "");
}

function linkIfKnown(name, ctx) {
  if (!ctx.typedefIndex?.has(name) || name === ctx.selfName) return null;
  return linkKnownName(name, name, ctx);
}

function linkKnownName(name, label, ctx) {
  const moduleName = ctx.typedefIndex?.get(name);
  if (!moduleName || name === ctx.selfName || !ctx.renderedNames?.has(name)) return null;
  const anchor = apiSymbolAnchor(moduleName, name);
  return `[\`${label}\`](${moduleHref(ctx.moduleName, moduleName)}#${anchor})`;
}

function linkCallable(ref, label, ctx) {
  const link = ctx.callableLinks?.get(ref);
  if (!link) return null;
  return `[\`${label}\`](${moduleHref(ctx.moduleName, link.moduleName)}#${link.anchor})`;
}

function linkReference(raw, label, ctx) {
  const ref = parseCallableReference(raw);
  if (ref?.method) return linkCallable(`${ref.owner}.${ref.method}`, label, ctx) ?? linkKnownName(ref.owner, label, ctx);
  if (ref) return linkCallable(ref.owner, label, ctx) ?? linkKnownName(ref.owner, label, ctx);
  return linkKnownName(raw, label, ctx);
}

function moduleHref(fromModule, toModule) {
  let rel = path.posix.relative(path.posix.dirname(fromModule), toModule);
  if (!rel.startsWith(".")) rel = `./${rel}`;
  return `${rel}.md`;
}

function renderUtilityType(utility, ctx) {
  const target = renderCallableReference(utility.target, ctx) ?? `\`${prettifyTypeString(utility.target)}\``;
  const suffix = utility.index == null ? "" : `[\`${utility.index}\`]`;
  return `\`${utility.kind}\`<${target}>${suffix}`;
}

function renderCallableReference(raw, ctx) {
  const ref = parseCallableReference(raw);
  if (!ref) return null;
  if (ref.method) {
    const key = `${ref.owner}.${ref.method}`;
    const linked = linkCallable(key, key, ctx);
    if (linked) return linked;
    return linkKnownName(ref.owner, `${ref.owner}.${ref.method}`, ctx) ?? `\`${ref.owner}.${ref.method}\``;
  }
  return linkCallable(ref.owner, ref.owner, ctx) ?? linkIfKnown(ref.owner, ctx) ?? `\`${ref.owner}\``;
}

// Rewrite `Check extends X ? A : B` to `A | B`, recursively and inside generic args
// (`Promise<T extends X ? A[] : A>` -> `Promise<A[] | A>`). Source branch order is kept; duplicates
// are removed. Callers skip `infer` and mapped types.
function rewriteConditionals(raw) {
  const s = unwrapParens(raw.trim());
  if (!s.includes("extends")) return s;

  const cond = splitConditional(s);
  if (cond) {
    const parts = [];
    for (const branch of [cond.whenTrue, cond.whenFalse]) {
      for (const part of splitTopLevel(rewriteConditionals(branch), "|")) {
        const trimmed = part.trim();
        // A `never` branch contributes nothing to the set of possible values.
        if (!trimmed || trimmed === "never" || parts.includes(trimmed)) continue;
        parts.push(trimmed);
      }
    }
    return parts.length ? parts.join(" | ") : "never";
  }

  for (const sep of ["|", "&"]) {
    const parts = splitTopLevel(s, sep);
    if (parts.length > 1) return parts.map((p) => rewriteConditionals(p.trim())).join(` ${sep} `);
  }

  if (s.endsWith("[]")) {
    const inner = rewriteConditionals(s.slice(0, -2));
    // A rewritten union has to be parenthesized before `[]` binds to it.
    return splitTopLevel(inner, "|").length > 1 ? `(${inner})[]` : `${inner}[]`;
  }

  if (s.startsWith("(") && matchingBracket(s, 0, "(", ")") === s.length - 1) {
    return `(${rewriteConditionals(s.slice(1, -1))})`;
  }

  const generic = s.match(NAMED_GENERIC);
  if (generic && matchingBracket(s, s.indexOf("<"), "<", ">") === s.length - 1) {
    const args = splitTopLevel(generic[2], ",")
      .map((a) => rewriteConditionals(a.trim()))
      .join(", ");
    return `${generic[1]}<${args}>`;
  }

  return s;
}

// Drop parentheses wrapping an entire type; keeping them would leave `(A | B) | C` in the output.
function unwrapParens(s) {
  let out = s;
  while (out.startsWith("(") && matchingBracket(out, 0, "(", ")") === out.length - 1) {
    out = out.slice(1, -1).trim();
  }
  return out;
}

// Mapped types (`{[K in Keys]: V}`) have no readable expansion, so they stay gnarly.
const MAPPED_TYPE = /\[\s*\w+\s+in\s/;

// Indexed access (`Acc['length']`, `DIM[0]`): the condition depends on a value the union of
// branches can't express.
const INDEXED_ACCESS = /\[['"0-9]/;

// Bail on conditionals whose condition carries meaning the branches lack: a spread accumulator, an
// indexed access, or a self-reference (e.g. depth-recursive `NestArray`). A whole-branch `never` is
// fine; `rewriteConditionals` drops it.
function isRewritableConditional(s, selfName) {
  if (/\binfer\b/.test(s) || MAPPED_TYPE.test(s)) return false;
  if (s.includes("...") || INDEXED_ACCESS.test(s)) return false;
  if (selfName && new RegExp(`\\b${selfName}\\b`).test(s)) return false;
  return true;
}

// A `never` left in the rewritten string was never a branch of its own: it sits inside a type
// argument, where the branch union can't explain it. Keep the original so it collapses.
function rewriteConditionalsSafely(s, selfName) {
  if (!isRewritableConditional(s, selfName)) return s;
  const rewritten = rewriteConditionals(s);
  return /\bnever\b/.test(rewritten) ? s : rewritten;
}

// Strip noisy TS constructs without rewriting structure. Conditional/mapped/infer types collapse to
// their outermost wrapper; unions of names and long generic lists survive so the renderer can link
// each part.
function prettifyTypeString(raw, selfName) {
  if (!raw) return "";
  let s = raw.trim();

  // Multi-line JSDoc types arrive with newlines; the scanners below assume single-line text.
  s = s.replace(/\s+/g, " ").trim();
  s = stripLeading(s, "<", ">"); // <T extends X>(...)
  s = stripImportPrefixes(s);
  s = s.replace(/import\(['"][^'"]+['"]\)/g, "any");
  s = s.replace(/"([^"\\]*(?:\\.[^"\\]*)*)"/g, "'$1'");

  if (parseUtilityType(s)) {
    return s;
  }

  s = rewriteConditionalsSafely(s, selfName);
  s = spaceObjectLiterals(s);

  if (isGnarly(s)) {
    const outer = s.match(/^([A-Za-z_$][\w$.]*)(?:<|\s|$)/);
    // A built-in utility type is meaningless shorn of its arguments (a naked `Parameters`).
    if (outer && TS_UTILITY_NAMES.has(outer[1])) return "unknown";
    // Keep the wrapper: a bare `Promise` is misleading, so emit `Promise<unknown>`.
    if (outer && /^[A-Za-z_$][\w$.]*</.test(s)) return `${outer[1]}<unknown>`;
    return outer ? outer[1] : "object";
  }
  return s.replace(/\s+/g, " ").trim();
}

// `{a: X}` -> `{ a: X }`. Empty `{}` and already-spaced braces are left alone.
function spaceObjectLiterals(s) {
  return s.replace(/\{(?![\s}])/g, "{ ").replace(/(?<![\s{])\}/g, " }");
}

function stripLeading(s, open, close) {
  const end = matchingBracket(s, 0, open, close);
  return end === -1 ? s : s.slice(end + 1).trim();
}

// Beyond this a type stops being scannable in a bullet or table cell, and collapses to its
// outermost wrapper.
const MAX_INLINE_TYPE_LENGTH = 160;

// Unreadable inline: mapped types, `infer`, an `extends` that could not be rewritten, indexed
// access into typeof, or anything over MAX_INLINE_TYPE_LENGTH. Callables and object literals are
// fine.
function isGnarly(s) {
  if (/\b(?:infer|extends)\b/.test(s)) return true;
  if (MAPPED_TYPE.test(s)) return true;
  // Any bracketed indexing (`X['foo']`, `X[0]`), including inside `Parameters<...>`, is too noisy
  // to render verbatim.
  if (/\[['"0-9]/.test(s)) return true;
  return s.length > MAX_INLINE_TYPE_LENGTH;
}
