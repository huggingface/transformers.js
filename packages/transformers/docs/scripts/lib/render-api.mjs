// Render a module's IR into readable markdown. Principles:
// - Prefer typedef names over their expanded TS structure
// - Never emit raw HTML tables or `&lt;code&gt;` escaping
// - Filter to the library's public export surface

export function renderModule(mod, ir, opts = {}) {
  const ctx = { typedefIndex: ir.typedefIndex, moduleName: mod.name };
  const publicNames = opts.publicNames ?? null;

  const classes = filterPublic(mod.classes, publicNames);
  const functions = filterPublic(mod.functions, publicNames);
  const constants = dedupeByName(filterPublic(mod.constants, publicNames));

  const out = [];
  out.push(`# ${mod.name}`, "");
  if (mod.description) out.push(mod.description.trim(), "");
  for (const ex of mod.examples) out.push(...renderExample(ex));

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
    const publicTypedefs = mod.typedefs.filter((td) => !isInternalTypedef(td));
    const rendered = publicTypedefs.flatMap((td) => renderTypedef(td, ctx));
    if (rendered.length) out.push("## Type Definitions", "", ...rendered);
  }
  if (mod.callbacks.length) {
    out.push("## Callbacks", "");
    for (const cb of mod.callbacks) out.push(...renderCallback(cb, ctx));
  }

  return (
    expandInlineLinks(out.join("\n"))
      .replace(/\n{3,}/g, "\n\n")
      .trimEnd() + "\n"
  );
}

function filterPublic(items, publicNames) {
  return publicNames ? items.filter((it) => publicNames.has(it.name)) : items;
}

function dedupeByName(items) {
  const seen = new Set();
  return items.filter((it) => (seen.has(it.name) ? false : seen.add(it.name)));
}

function renderExample(ex) {
  const lines = [];
  if (ex.title) lines.push(`**Example:** ${ex.title}`);
  lines.push("```" + ex.language, ex.code, "```", "");
  return lines;
}

// Descriptions carried over from the Python library sometimes start with a
// reST-style `[`TypeName`]` cross-reference that JavaScript readers can't follow.
// Strip the artifact from the start of the first paragraph; leave the rest alone.
// Also normalize `@see Symbol` / `@see {@link Symbol}` to inline markup so it
// reads as prose rather than raw JSDoc tags — `{@link ...}` is expanded later
// by `expandInlineLinks`.
function cleanDescription(text) {
  return text
    .trim()
    .replace(/^\[`?([A-Za-z_$][\w$.]*)`?\]\s*/, "")
    .replace(/@see\s+(?=\{@link)/g, "")
    .replace(/@see\s+`?([A-Za-z_$][\w$.]*)`?/g, "`$1`");
}

// `{@link url}` / `{@link url Text}` / `{@link Symbol}` -> markdown.
function expandInlineLinks(text) {
  return text.replace(/\{@link\s+([^}\s]+)(?:\s+([^}]+))?\}/g, (_, target, label) => {
    const displayed = (label ?? target).trim();
    return /^https?:\/\//.test(target) ? `[${displayed}](${target})` : `\`${displayed}\``;
  });
}

// ---------- classes, functions, callbacks ----------

function renderClass(cls, ctx) {
  const lines = [`### ${cls.name}`, ""];
  if (cls.description) lines.push(cleanDescription(cls.description), "");
  for (const ex of cls.examples) lines.push(...renderExample(ex));

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
  // Skip undocumented, untyped field entries — pure placeholders with no
  // reader value. Deprecation flags keep a field visible as a warning.
  if (!f.type && !f.description && !f.deprecated && f.defaultValue == null) return [];

  const type = f.type ? ` : ${renderType(f.type, ctx)}` : "";
  const lines = [`#### \`${parent}.${f.name}\`${type}`, ""];
  if (f.deprecated) lines.push("> **Deprecated**", "");
  if (f.description) lines.push(cleanDescription(f.description), "");
  if (f.defaultValue != null && f.defaultValue !== "") lines.push(`**Default:** \`${f.defaultValue}\``, "");
  return lines;
}

// `_`-prefixed methods are the library's convention for internal / subclass-only
// hooks — they're not part of the user-facing API. Exception: `constructor` is
// kept even though it has no leading underscore.
function shouldRenderMethod(m) {
  if (m.name.startsWith("_")) return false;
  return m.description || m.params?.length || m.returns?.description || m.returns?.type || m.examples?.length || m.throws?.length;
}

function renderFunction(fn, ctx, depth, parent = null) {
  const lines = [`${"#".repeat(depth)} ${signature(fn, parent)}`, ""];
  // Resolve generic type parameters (`@template {Constraint} T`) inside this
  // function's parameter/return types. Without the constraint map, a `T`
  // would render as `any`.
  const templateMap = new Map();
  for (const t of fn.templates ?? []) if (t.name && t.type) templateMap.set(t.name, t.type);
  const fnCtx = templateMap.size ? { ...ctx, templates: templateMap } : ctx;

  if (fn.deprecated) lines.push("> **Deprecated**", "");
  if (fn.description) lines.push(cleanDescription(fn.description), "");
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
  for (const ex of fn.examples) lines.push(...renderExample(ex));
  return lines;
}

function signature(fn, parent) {
  const params = (fn.params || [])
    .filter((p) => p.name && !p.name.includes("."))
    .map((p) => (p.optional ? `[${p.name}]` : p.name))
    .join(", ");
  const owner = parent ? `${parent}.` : "";
  return `\`${owner}${fn.name}(${params})\``;
}

function renderCallback(cb, ctx) {
  const lines = [`### ${cb.name}`, ""];
  if (cb.description) lines.push(cleanDescription(cb.description), "");
  if (cb.params?.length) {
    lines.push("**Parameters**", "", ...renderParamList(cb.params, ctx), "");
  }
  if (cb.returns?.type || cb.returns?.description) {
    const type = cb.returns.type ? renderType(cb.returns.type, ctx) : "";
    const desc = cb.returns.description ? ` — ${cb.returns.description}` : "";
    lines.push(`**Returns:** ${type}${desc}`, "");
  }
  return lines;
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
  const desc = p.description ? ` — ${indentContinuations(p.description.trim(), contPad)}` : "";
  const line = `${pad}- \`${simpleName(p.name)}\`${type}${opt}${def}${desc}`;
  return [line, ...p.children.flatMap((c) => renderParamNode(c, ctx, indent + 1))];
}

// Indent every line after the first so multi-line descriptions (including
// bulleted continuations) render inside their parent list item.
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
  const lines = [`### \`${c.name}\`${type}`, ""];
  if (c.description) lines.push(cleanDescription(c.description), "");
  for (const ex of c.examples) lines.push(...renderExample(ex));
  return lines;
}

// Skip typedefs that exist purely to thread generic parameters through the
// type system (`@typedef {T} Name`, `_`-prefixed names) or that are opaque
// placeholders with no useful content to show a reader.
function isInternalTypedef(td) {
  if (td.name.startsWith("_")) return true;
  const type = (td.type ?? "").trim();
  // `@typedef {object} Foo` with no body — nothing to show.
  if ((type === "object" || type === "Object") && !td.description && !td.properties?.length) return true;
  // `@typedef {GenericParam} Foo` — bare alias with no description or body.
  if (/^[A-Za-z_$][\w$.]*$/.test(type) && !td.description && !td.properties?.length) return true;
  return false;
}

// Generic type parameter names (T, K, V, TItem, TReturnTensor, ...) — show as
// `any` in rendered types so the reader doesn't chase an undefined symbol.
function isGenericParamName(name) {
  return /^T[A-Z][A-Za-z]*$/.test(name) || /^[TKV]$/.test(name);
}

function renderTypedef(td, ctx) {
  const displayed = td.type ? renderType(td.type, { ...ctx, selfName: td.name }) : "";
  const isSelfReference = displayed === `\`${td.name}\``;
  const isGenericPassthrough = /^`[A-Z][A-Za-z]?`$/.test(displayed);
  // Collapsed fallbacks (`object`, `unknown`, `any`) carry no real information
  // — showing `_Type:_ `object`` adds noise without helping the reader.
  const isOpaque = displayed === "`object`" || displayed === "`unknown`" || displayed === "`any`";
  // Unions and intersections are worth showing even when long — every variant
  // is a named type the reader can click through to. Opaque inline object
  // literals (`{...}`) stay hidden.
  const isUnionOrIntersection = / \| | & /.test(displayed);
  const fitsInline = displayed.length < 120;
  const typeIsShowable =
    displayed && (isUnionOrIntersection || fitsInline) && !displayed.startsWith("`{") && !isSelfReference && !isGenericPassthrough && !isOpaque;

  // Don't emit an empty `### Name` heading. A typedef needs *something* the
  // reader can't derive from the name alone: a description, properties, or a
  // concise displayable type.
  if (!td.description && !td.properties?.length && !typeIsShowable) return [];

  const lines = [`### ${td.name}`, ""];
  if (td.description) lines.push(cleanDescription(td.description), "");
  if (typeIsShowable) {
    lines.push(`_Type:_ ${displayed}`, "");
  }
  if (td.properties?.length) {
    lines.push("**Properties**", "", ...renderParamList(td.properties, ctx), "");
  }
  return lines;
}

// ---------- type rendering ----------

const GENERIC_WRAPPERS = /^(Promise|Array|Record|Map|Set|Iterable|AsyncIterable|Partial|Readonly)<(.+)>$/;
const SIMPLE_NAME = new RegExp(`^[A-Za-z_$][\\w$.]*$`);
const INDEXED_NAME = /^([A-Za-z_$][\w$.]*)\[[^\]]+\]$/;

// Turn a raw JSDoc type string into readable markdown. Prefers author-written
// names over expanded TS structures; unknown gnarly types become `object`.
export function renderType(raw, ctx, opts = {}) {
  const pretty = prettifyTypeString(raw);
  if (opts.noLink) return `\`${pretty}\``;

  // Unions split first. `_`-prefixed variants (internal types from intersections)
  // are filtered out so they don't leak into the public docs.
  const unionParts = splitTopLevel(pretty, "|")
    .map((p) => p.trim())
    .filter((p) => !/^_[A-Za-z]/.test(p));
  if (unionParts.length > 1) return unionParts.map((p) => renderType(p, ctx)).join(" | ");
  if (unionParts.length === 1 && unionParts[0] !== pretty.trim()) return renderType(unionParts[0], ctx);

  // Same for intersections (`A & B`): drop internal variants before joining.
  const intersectParts = splitTopLevel(pretty, "&")
    .map((p) => p.trim())
    .filter((p) => !/^_[A-Za-z]/.test(p));
  if (intersectParts.length > 1) return intersectParts.map((p) => renderType(p, ctx)).join(" & ");
  if (intersectParts.length === 1 && intersectParts[0] !== pretty.trim()) return renderType(intersectParts[0], ctx);

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

  return `\`${pretty}\``;
}

function linkIfKnown(name, ctx) {
  if (!ctx.typedefIndex?.has(name) || name === ctx.selfName) return null;
  const anchor = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `[\`${name}\`](./${ctx.typedefIndex.get(name)}.md#${anchor})`;
}

// Split `text` on `sep`, ignoring separators inside brackets, braces, parens,
// angle brackets, or string literals.
function splitTopLevel(text, sep) {
  const out = [];
  let depth = 0;
  let inStr = null;
  let buf = "";
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === inStr && text[i - 1] !== "\\") inStr = null;
      buf += ch;
    } else if (ch === '"' || ch === "'" || ch === "`") {
      inStr = ch;
      buf += ch;
    } else if ("<({[".includes(ch)) {
      depth++;
      buf += ch;
    } else if (">)}]".includes(ch)) {
      depth--;
      buf += ch;
    } else if (depth === 0 && ch === sep) {
      out.push(buf);
      buf = "";
    } else {
      buf += ch;
    }
  }
  if (buf) out.push(buf);
  return out.length > 1 ? out : [text];
}

// Strip noisy TS constructs from a type string without rewriting structure.
// Conditional/mapped/infer types collapse to their outermost wrapper; simple
// unions of names or long generic lists are preserved so the renderer can
// split them into individual links.
const TS_UTILITY_NAMES = new Set(["Parameters", "ReturnType", "ConstructorParameters", "InstanceType", "ThisType"]);
export function prettifyTypeString(raw) {
  if (!raw) return "";
  let s = raw.trim();

  s = stripLeading(s, "<", ">"); // <T extends X>(...)
  s = s.replace(/import\(['"][^'"]+['"]\)\.([A-Za-z_$][\w$.]*)/g, "$1");
  s = s.replace(/import\(['"][^'"]+['"]\)/g, "any");

  if (isGnarly(s)) {
    const outer = s.match(/^([A-Za-z_$][\w$.]*)(?:<|\s|$)/);
    // Built-in TS utility types are meaningless shorn of their arguments — the
    // reader is better served by `unknown` than by a naked `Parameters`.
    if (outer && TS_UTILITY_NAMES.has(outer[1])) return "unknown";
    return outer ? outer[1] : "object";
  }
  return s.replace(/\s+/g, " ").trim();
}

function stripLeading(s, open, close) {
  if (s[0] !== open) return s;
  let depth = 1;
  let i = 1;
  while (i < s.length && depth > 0) {
    if (s[i] === open) depth++;
    else if (s[i] === close) depth--;
    i++;
  }
  return depth === 0 ? s.slice(i).trim() : s;
}

// Conditional types (`A extends B ? X : Y`), mapped types (`{ [K in ...] }`),
// and `infer` are unreadable inline. So are indexed accesses into typeof
// expressions (`Parameters<X['foo']>[0]`) — readable names these are not.
function isGnarly(s) {
  if (/\b(?:infer|extends)\b/.test(s)) return true;
  // Any bracketed indexing like `X['foo']` or `X[0]` inside the type string —
  // including inside `Parameters<...>` / `ReturnType<...>` — makes it too
  // noisy to render verbatim.
  if (/\[['"0-9]/.test(s)) return true;
  let angle = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "<") angle++;
    else if (s[i] === ">") angle = Math.max(0, angle - 1);
    else if (angle === 0 && (s[i] === "?" || (s[i] === "[" && s[i + 1] === "K"))) return true;
  }
  return false;
}
