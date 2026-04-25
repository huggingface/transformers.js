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

const GENERATED_BANNER = "<!-- DO NOT EDIT: generated from src/**/*.js by docs/scripts/generate-all.js -->";
const DOCS_SITE = "https://huggingface.co/docs/transformers.js/api";

export function renderSkill({ ir, tasks, publicNames, skillDir }) {
  const ctx = { ir, tasks, publicNames: publicNames ?? null };

  // Rewrite fully-generated reference files.
  const refDir = path.join(skillDir, "references");
  fs.mkdirSync(refDir, { recursive: true });
  fs.writeFileSync(path.join(refDir, "TASKS.md"), absolutize(renderTasks(ctx)));

  // Expand `<!-- @generated:start id=... -->` markers in every hand-written
  // markdown file under the skill directory. Prose outside markers is preserved.
  // Only the generated blocks get absolutized — hand-authored prose is left alone.
  for (const file of walkMarkdown(skillDir)) {
    const original = fs.readFileSync(file, "utf8");
    if (!original.includes("@generated:start")) continue;
    const injected = injectMarkers(original, ctx);
    if (injected !== original) fs.writeFileSync(file, injected);
  }
}

// Skill files live outside the docs site, so relative `./foo.md#bar` links —
// inherited from JSDoc descriptions and from our typedef cross-reference
// renderer — have to be rewritten to the public docs URL. `.md` becomes the
// extensionless path the site uses.
function absolutize(markdown) {
  return markdown.replace(/\]\(\.\/([^)\s]+?)\.md(#[^)\s]*)?\)/g, (_, page, anchor = "") => {
    return `](${DOCS_SITE}/${page}${anchor})`;
  });
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
    return `${startTag}\n${absolutize(content).trimEnd()}\n${endTag}`;
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
    lines.push(`- [\`${taskId}\`](references/TASKS.md#${taskAnchor(taskId)})${aliasSuffix} — default model: \`${info.defaultModel}\``);
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
    const type = p.type ? renderTypedefType(p.type, { table: true }) : "";
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
      lines.push(`- \`${f.name}\`${type}${f.description ? ` — ${firstSentence(f.description)}` : ""}`);
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
      lines.push(`- \`${m.name}(${params})\`${ret} — ${firstSentence(m.description)}`);
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
function renderTypedefType(raw, { table = false } = {}) {
  let compact = raw
    .replace(/import\(['"][^'"]+['"]\)\.([A-Za-z_$][\w$.]*)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  if (compact.length > 60 && (compact.startsWith("{") || /[({=]/.test(compact))) {
    compact = "object";
  }
  const escaped = compact.replace(table ? /[`|]/g : /`/g, (ch) => `\\${ch}`);
  return "`" + escaped + "`";
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

// Per-task recipe page. Grouped by modality so readers can find the task they
// want quickly, with a table of contents up front.
function renderTasks(ctx) {
  const lines = [
    GENERATED_BANNER,
    "",
    "# Tasks",
    "",
    "Runnable recipes for every task exposed through the `pipeline()` API, grouped by modality.",
    "Each section is pulled from the pipeline class's JSDoc, so the examples stay in sync with the library.",
    "",
    "## Contents",
    "",
  ];
  const groups = groupTasksByModality(ctx.tasks.supportedTasks);
  for (const [group, ids] of groups) {
    lines.push(`**${group}** — ${ids.map((id) => `[\`${id}\`](#${taskAnchor(id)})`).join(" · ")}`, "");
  }
  for (const [group, ids] of groups) {
    lines.push(`## ${group}`, "");
    for (const id of ids) {
      const info = ctx.tasks.supportedTasks.get(id);
      lines.push(`### \`${id}\``, "", renderTaskRecipe(id, info, ctx), "");
    }
  }
  return finalize(lines);
}

function taskAnchor(id) {
  return id.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

// Keyword-based grouping keeps the TOC stable without hard-coding per-task
// metadata. A task falls into the first modality whose keyword matches its id.
// Order matters: `text-to-audio` must hit "Audio" before "Text".
const MODALITY_RULES = [
  ["Audio", /audio|speech|asr|tts/],
  ["Vision", /image|vision|depth|object|segment|background|document/],
  ["Text", /text|translation|summar|generation|question|fill|mask|classification|ner/],
  ["Embeddings", /feature|embedding/],
];

function groupTasksByModality(supportedTasks) {
  const groups = new Map(MODALITY_RULES.map(([name]) => [name, []]));
  groups.set("Other", []);
  for (const id of supportedTasks.keys()) {
    const group = MODALITY_RULES.find(([, re]) => re.test(id))?.[0] ?? "Other";
    groups.get(group).push(id);
  }
  for (const [name, ids] of [...groups]) if (!ids.length) groups.delete(name);
  return groups;
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

// Extract the first complete sentence. A line break inside a paragraph is not
// a sentence boundary, so we re-flow a paragraph onto one line before cutting.
function firstSentence(text) {
  if (!text) return "_(undocumented)_";
  const paragraph = text
    .split(/\n\s*\n/, 1)[0]
    .replace(/\s+/g, " ")
    .trim();
  const sanitized = stripDocArtifacts(paragraph);
  const match = sanitized.match(/^(.+?[.!?])(?=\s|$)/);
  const sentence = (match ? match[1] : sanitized).trim();
  return sentence.endsWith(".") || sentence.endsWith("!") || sentence.endsWith("?") ? sentence : sentence + ".";
}

// Descriptions copied from the Python library sometimes start with
// `[`TypeName`]` (reST cross-reference syntax). Drop the leading artifact.
function stripDocArtifacts(text) {
  return text.replace(/^\[`?[A-Za-z_$][\w$.]*`?\]\s*/, "");
}
