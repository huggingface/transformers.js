// Plain-text helpers shared between the API and skill renderers.

// First complete sentence of a description. A line break inside a paragraph is not a sentence
// boundary, so the paragraph is re-flowed onto one line before cutting. Empty input returns "".
export function firstSentence(text) {
  if (!text) return "";
  const paragraph = stripDocArtifacts(
    text
      .split(/\n\s*\n/, 1)[0]
      .replace(/\s+/g, " ")
      .trim(),
  );
  const match = paragraph.match(/^(.+?[.!?])(?=\s|$)/);
  const sentence = (match ? match[1] : paragraph).trim();
  return /[.!?]$/.test(sentence) ? sentence : sentence + ".";
}

// reST cross-references inherited from the Python library: ``[`TypeName`]``, sometimes `~`-prefixed
// or `()`-suffixed. Renderers that can resolve anchors link them; the rest reduce them to inline
// code. A markdown link `[label](url)` keeps its load-bearing bracket.
export const DOC_REFERENCE = /\[`~?([A-Za-z_$][\w$.]*(?:\(\))?)`\](?!\()/g;

export function stripDocArtifacts(text) {
  return text.replace(DOC_REFERENCE, "`$1`");
}

const FENCE = /^\s*```/;

// `fenced` is true for lines inside a fenced code block, including the delimiters. Line structure is
// preserved; an unterminated fence extends to the end of the text.
export function mapLines(text, fn) {
  let inFence = false;
  return text
    .split("\n")
    .map((line) => {
      const isFence = FENCE.test(line);
      if (isFence) inFence = !inFence;
      return fn(line, isFence || inFence);
    })
    .join("\n");
}

// Each fence is swapped for a single-line placeholder while `transform` runs, so page-level
// rewrites can't mangle example code. An unterminated fence extends to the end of the text.
export function transformOutsideFences(text, transform) {
  const blocks = [];
  const out = [];
  let fence = null;
  for (const line of text.split("\n")) {
    if (fence) {
      fence.push(line);
      if (FENCE.test(line)) {
        out.push(`\u0000${blocks.length}\u0000`);
        blocks.push(fence.join("\n"));
        fence = null;
      }
    } else if (FENCE.test(line)) {
      fence = [line];
    } else {
      out.push(line);
    }
  }
  if (fence) {
    out.push(`\u0000${blocks.length}\u0000`);
    blocks.push(fence.join("\n"));
  }
  return transform(out.join("\n")).replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

// Comma-separated parameter names for a signature. Only top-level params: `options.foo` rows are
// nested options covered by the linked typedef. Optional params render as `[name]`.
export function paramSignature(params) {
  return (params ?? [])
    .filter((p) => p.name && !p.name.includes("."))
    .map((p) => (p.optional ? `[${p.name}]` : p.name))
    .join(", ");
}

// One `**Example:** <title>` block with its fenced code, as markdown lines. `blankAfterTitle` adds
// the blank line module-example markers use.
export function exampleLines(ex, { blankAfterTitle = false } = {}) {
  const lines = [];
  if (ex.title) {
    lines.push(`**Example:** ${ex.title}`);
    if (blankAfterTitle) lines.push("");
  }
  lines.push("```" + ex.language, ex.code, "```", "");
  return lines;
}

// `import('./module.js').Name` -> `Name`, for display.
export function stripImportPrefixes(type) {
  return type.replace(/import\(['"][^'"]+['"]\)\.([A-Za-z_$][\w$.]*)/g, "$1");
}

// Split a type that is exactly one `import('./module.js').Name` into specifier and name; else null.
export function parseImportedType(type) {
  const m = type?.trim().match(/^import\(['"]([^'"]+)['"]\)\.([A-Za-z_$][\w$]*)$/);
  return m ? { specifier: m[1], name: m[2] } : null;
}
