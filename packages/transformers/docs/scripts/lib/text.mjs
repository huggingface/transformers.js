// Plain-text helpers shared between the API and skill renderers.

// Extract the first complete sentence of a description. A line break inside a
// paragraph is not a sentence boundary, so re-flow the paragraph onto one line
// before cutting. Returns an empty string when given empty input — callers are
// expected to skip rendering rather than emit a placeholder.
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

// Descriptions copied from the Python library sometimes start with
// `[`TypeName`]` (reST cross-reference syntax). Drop the leading artifact —
// but not a markdown link `[label](url)`, whose bracket is load-bearing.
export function stripDocArtifacts(text) {
  return text.replace(/^\[`?[A-Za-z_$][\w$.]*`?\](?!\()\s*/, "");
}

// Run `transform` over the parts of `text` outside fenced code blocks. Each
// fence is swapped for a single-line placeholder while the transform runs, so
// page-level rewrites (link expansion, blank-run collapsing) can't mangle
// example code. An unterminated fence extends to the end of the text.
export function transformOutsideFences(text, transform) {
  const FENCE = /^\s*```/;
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

// One `**Example:** <title>` block with its fenced code, as markdown lines.
// `blankAfterTitle` keeps the historical spacing of module-example markers.
export function exampleLines(ex, { blankAfterTitle = false } = {}) {
  const lines = [];
  if (ex.title) {
    lines.push(`**Example:** ${ex.title}`);
    if (blankAfterTitle) lines.push("");
  }
  lines.push("```" + ex.language, ex.code, "```", "");
  return lines;
}

// `import('./module.js').Name` prefixes appear throughout JSDoc type strings.
// Strip them down to the bare (possibly dotted) name for display.
export function stripImportPrefixes(type) {
  return type.replace(/import\(['"][^'"]+['"]\)\.([A-Za-z_$][\w$.]*)/g, "$1");
}

// Parse a type that is exactly one `import('./module.js').Name` expression
// into its module specifier and imported name; null for anything else.
export function parseImportedType(type) {
  const m = type?.trim().match(/^import\(['"]([^'"]+)['"]\)\.([A-Za-z_$][\w$]*)$/);
  return m ? { specifier: m[1], name: m[2] } : null;
}
