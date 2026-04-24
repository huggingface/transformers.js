// Render the skill files in `.ai/skills/transformers-js/`. Two kinds of output:
//
//  1. Generated fragments injected into hand-written `SKILL.md` between
//     `<!-- @generated:start id=X -->` and `<!-- @generated:end id=X -->`
//     sentinels. Prose outside the markers is preserved across runs.
//
//  2. Fully generated files under `references/`. These carry a banner at the
//     top and are rewritten in full on every run.

import fs from "node:fs";
import path from "node:path";

const GENERATED_BANNER = "<!-- DO NOT EDIT: generated from src/**/*.js by docs/scripts/generate-skill.js -->";

export function renderSkill({ ir, tasks, publicNames, skillDir }) {
  const ctx = { ir, tasks, publicNames: publicNames ?? null };

  // Rewrite fully-generated reference files.
  const refDir = path.join(skillDir, "references");
  fs.mkdirSync(refDir, { recursive: true });
  fs.writeFileSync(path.join(refDir, "TASK_EXAMPLES.md"), renderTaskExamples(ctx));
  fs.writeFileSync(path.join(refDir, "API_SUMMARY.md"), renderApiSummary(ctx));

  // Expand `<!-- @generated:start id=... -->` markers in every hand-written
  // markdown file under the skill directory. Prose outside markers is preserved.
  for (const file of walkMarkdown(skillDir)) {
    const original = fs.readFileSync(file, "utf8");
    if (!original.includes("@generated:start")) continue;
    const injected = injectMarkers(original, ctx);
    if (injected !== original) fs.writeFileSync(file, injected);
  }
}

function walkMarkdown(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walkMarkdown(full));
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

// ---------- marker injection ----------

const MARKER_RE = /(<!-- @generated:start id=([^\s]+) -->)([\s\S]*?)(<!-- @generated:end id=\2 -->)/g;

function injectMarkers(original, ctx) {
  return original.replace(MARKER_RE, (match, startTag, id, _body, endTag) => {
    const content = resolveMarker(id, ctx);
    if (content == null) {
      console.warn(`skill: unknown marker id "${id}"`);
      return match;
    }
    return `${startTag}\n${content.trimEnd()}\n${endTag}`;
  });
}

// Dispatch a marker id to its generator. Supported ids:
//   task-list                  full list of supported pipeline tasks
//   task:<id>                  recipe for a single task
//   typedef:<Name>             properties table for a named typedef
//   class:<Name>               class description + fields + methods summary
//   fields:<ClassName>         properties table built from a class's fields
//   examples:<module>          all inline `**Example:**` blocks from `@module <module>`
function resolveMarker(id, ctx) {
  if (id === "task-list") return renderTaskList(ctx);

  const [kind, arg] = splitOnce(id, ":");
  switch (kind) {
    case "task": {
      const info = ctx.tasks.supportedTasks.get(arg);
      return info ? renderTaskRecipe(arg, info, ctx) : null;
    }
    case "typedef":
      return renderTypedefTable(arg, ctx);
    case "class":
      return renderClassSummary(arg, ctx);
    case "fields":
      return renderFieldsTable(arg, ctx);
    case "examples":
      return renderModuleExamples(arg, ctx);
    default:
      return null;
  }
}

function renderModuleExamples(moduleName, ctx) {
  const mod = ctx.ir.modules.find((m) => m.name === moduleName);
  if (!mod) {
    console.warn(`skill: module "${moduleName}" not found`);
    return "";
  }
  if (!mod.examples.length) {
    console.warn(`skill: module "${moduleName}" has no example blocks`);
    return "";
  }
  const lines = [];
  for (const ex of mod.examples) {
    if (ex.title) lines.push(`**Example:** ${ex.title}`, "");
    lines.push("```" + ex.language, ex.code, "```", "");
  }
  return lines.join("\n").trimEnd();
}

function renderFieldsTable(className, ctx) {
  const cls = findClass(ctx.ir, className);
  if (!cls) {
    console.warn(`skill: class "${className}" not found`);
    return "";
  }
  const fields = cls.members.filter((m) => m.kind === "field");
  return (
    propertiesTable(
      fields.map((f) => ({
        name: f.name,
        type: f.type,
        optional: f.defaultValue != null, // class fields with initializers are effectively optional
        defaultValue: f.defaultValue,
        description: f.description,
      })),
    ) || warnEmpty(className)
  );
}

function splitOnce(s, sep) {
  const i = s.indexOf(sep);
  return i < 0 ? [s, ""] : [s.slice(0, i), s.slice(i + 1)];
}

// ---------- generated content ----------

function renderTaskList({ tasks }) {
  const lines = [];
  for (const [taskId, info] of tasks.supportedTasks) {
    const aliases = aliasesFor(taskId, tasks);
    const aliasSuffix = aliases.length ? ` _(alias: ${aliases.map((a) => `\`${a}\``).join(", ")})_` : "";
    lines.push(`- \`${taskId}\`${aliasSuffix} — default model: \`${info.defaultModel}\``);
  }
  return lines.join("\n");
}

function renderTaskRecipe(taskId, info, ctx) {
  const cls = findClass(ctx.ir, info.pipelineClass);
  const aliases = aliasesFor(taskId, ctx.tasks);
  const lines = [`**Default model:** \`${info.defaultModel}\``];
  if (aliases.length) lines.push(`**Aliases:** ${aliases.map((a) => `\`${a}\``).join(", ")}`);
  lines.push("");
  if (cls?.description) lines.push(cls.description.trim(), "");
  for (const ex of cls?.examples ?? []) {
    if (ex.title) lines.push(`**Example:** ${ex.title}`);
    lines.push("```" + ex.language, ex.code, "```", "");
  }
  return lines.join("\n").trimEnd();
}

function renderTypedefTable(name, ctx) {
  const td = findTypedef(ctx.ir, name);
  if (!td) {
    console.warn(`skill: typedef "${name}" not found`);
    return "";
  }

  // Discriminated union (`A | B | C`): render one table per variant.
  const variants = splitUnion(td.type);
  if (variants.length > 1) {
    const parts = [];
    for (const variantName of variants) {
      const table = propertiesTable(collectProperties(variantName, ctx.ir));
      if (!table) continue;
      parts.push(`**\`${variantName}\`**`, "", table, "");
    }
    return parts.length ? parts.join("\n").trimEnd() : warnEmpty(name);
  }

  const props = collectProperties(name, ctx.ir);
  return propertiesTable(props) || warnEmpty(name);
}

function warnEmpty(name) {
  console.warn(`skill: typedef "${name}" has no properties to render`);
  return "";
}

function propertiesTable(props) {
  if (!props.length) return "";
  const lines = ["| Option | Type | Description |", "|--------|------|-------------|"];
  for (const p of props) {
    const type = p.type ? renderTypedefType(p.type) : "";
    const nameCell = p.optional ? `\`${p.name}\`?` : `\`${p.name}\``;
    const desc = prepareCell(p.description) + (p.defaultValue != null ? ` _(default: \`${p.defaultValue}\`)_` : "");
    lines.push(`| ${nameCell} | ${type} | ${desc} |`);
  }
  return lines.join("\n");
}

// Split a type string on top-level `|`, returning just the referenced names.
// Returns [] if the type doesn't look like a union of simple names.
function splitUnion(type) {
  if (!type) return [];
  const parts = [];
  let depth = 0;
  let buf = "";
  for (const ch of type) {
    if ("<({[".includes(ch)) depth++;
    else if (">)}]".includes(ch)) depth--;
    if (depth === 0 && ch === "|") {
      parts.push(buf.trim());
      buf = "";
    } else {
      buf += ch;
    }
  }
  if (buf.trim()) parts.push(buf.trim());
  if (parts.length < 2) return [];
  if (!parts.every((p) => /^[A-Za-z_$][\w$.]*$/.test(p))) return [];
  return parts;
}

function renderClassSummary(name, ctx) {
  const cls = findClass(ctx.ir, name);
  if (!cls) {
    console.warn(`skill: class "${name}" not found`);
    return "";
  }
  const lines = [];
  if (cls.description) lines.push(cls.description.trim(), "");
  const fields = cls.members.filter((m) => m.kind !== "method");
  const methods = cls.members.filter((m) => m.kind === "method" && m.description);
  if (fields.length) {
    lines.push("**Fields**", "");
    for (const f of fields) {
      const type = f.type ? ` (\`${f.type}\`)` : "";
      lines.push(`- \`${f.name}\`${type}${f.description ? ` — ${firstLine(f.description)}` : ""}`);
    }
    lines.push("");
  }
  if (methods.length) {
    lines.push("**Methods**", "");
    for (const m of methods) {
      // Only show top-level params in the summary signature — `options.foo`
      // rows are nested options, covered by the linked typedef.
      const params = (m.params ?? [])
        .filter((p) => p.name && !p.name.includes("."))
        .map((p) => (p.optional ? `[${p.name}]` : p.name))
        .join(", ");
      const ret = m.returns?.type ? ` → \`${prettifyReturnType(m.returns.type)}\`` : "";
      lines.push(`- \`${m.name}(${params})\`${ret} — ${firstLine(m.description)}`);
    }
    lines.push("");
  }
  return lines.join("\n").trimEnd();
}

function prettifyReturnType(raw) {
  return raw
    .replace(/import\(['"][^'"]+['"]\)\.([A-Za-z_$][\w$.]*)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

// Flatten a typedef into its effective properties. If the typedef's type is
// an intersection like `A & B`, collect properties from each referenced
// typedef in declaration order (de-duplicated by name).
function collectProperties(name, ir) {
  const typedef = findTypedef(ir, name);
  if (!typedef) return [];
  if (typedef.properties?.length) return typedef.properties;

  const parts = (typedef.type ?? "")
    .split("&")
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length < 2) return [];

  const seen = new Map();
  for (const part of parts) {
    for (const p of collectProperties(part, ir)) {
      if (!seen.has(p.name)) seen.set(p.name, p);
    }
  }
  return [...seen.values()];
}

// Use the IR's cross-reference index (which already prefers canonical
// definitions over re-export aliases) to find the authoritative typedef.
function findTypedef(ir, name) {
  const modName = ir.typedefIndex.get(name);
  if (!modName) return null;
  const mod = ir.modules.find((m) => m.name === modName);
  return mod?.typedefs.find((t) => t.name === name) ?? null;
}

// Compact display: inside a markdown table cell we want inline code spans.
// Collapse `import('./x.js').Foo` to `Foo`, long inline object types to
// `object`, and strip newlines.
function renderTypedefType(raw) {
  let compact = raw
    .replace(/import\(['"][^'"]+['"]\)\.([A-Za-z_$][\w$.]*)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  if (compact.length > 60 && (compact.startsWith("{") || /[({=]/.test(compact))) {
    compact = "object";
  }
  return "`" + compact.replace(/`/g, "\\`") + "`";
}

// Cells can't contain newlines or un-escaped pipes. `{@link url}` gets turned
// into a plain URL so agents can still see it.
function prepareCell(text) {
  return (text || "")
    .replace(/\{@link\s+([^}\s]+)(?:\s+[^}]+)?\}/g, "$1")
    .replace(/\|/g, "\\|")
    .replace(/\n+/g, " ")
    .trim();
}

function renderTaskExamples(ctx) {
  const lines = [
    GENERATED_BANNER,
    "",
    "# Task Examples",
    "",
    "Runnable recipes for every task supported by the `pipeline()` API, extracted",
    "directly from each pipeline class's JSDoc.",
    "",
  ];
  for (const [taskId, info] of ctx.tasks.supportedTasks) {
    lines.push(`## \`${taskId}\``, "", renderTaskRecipe(taskId, info, ctx), "");
  }
  return finalize(lines);
}

function renderApiSummary(ctx) {
  const lines = [GENERATED_BANNER, "", "# API Summary", "", "Condensed index of publicly exported classes and functions.", ""];
  for (const mod of ctx.ir.modules) {
    const classes = filterPublic(mod.classes, ctx.publicNames);
    const functions = filterPublic(mod.functions, ctx.publicNames);
    if (!classes.length && !functions.length) continue;
    lines.push(`## \`${mod.name}\``, "");
    for (const cls of classes) lines.push(`- **\`${cls.name}\`** — ${firstLine(cls.description)}`);
    for (const fn of functions) lines.push(`- \`${fn.name}()\` — ${firstLine(fn.description)}`);
    lines.push("");
  }
  return finalize(lines);
}

function finalize(lines) {
  return (
    lines
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trimEnd() + "\n"
  );
}

function filterPublic(items, publicNames) {
  return publicNames ? items.filter((it) => publicNames.has(it.name)) : items;
}

function aliasesFor(taskId, tasks) {
  return [...tasks.aliases.entries()].filter(([, t]) => t === taskId).map(([a]) => a);
}

// ---------- helpers ----------

function findClass(ir, name) {
  for (const mod of ir.modules) {
    const cls = mod.classes.find((c) => c.name === name);
    if (cls) return cls;
  }
  return null;
}

function firstLine(text) {
  if (!text) return "_(undocumented)_";
  const line = text.split("\n", 1)[0].trim();
  return line.endsWith(".") ? line : line + ".";
}
