// JSDoc tag parser. Given a raw `/** ... */` comment, returns
// `{ description, tags }`. The TS-compiler pass in structure.mjs decides
// which comments attach to which AST node; this module only parses content.

const TAG_START = /^@[A-Za-z]/;
const FENCE = /^```/;
const IDENT_PAT = /[A-Za-z_$][\w$]*/.source;
const DOTTED_PAT = /[A-Za-z_$][\w$.]*/.source;

export function parseJsDoc(raw) {
  if (!raw) return null;
  const segments = splitIntoSegments(stripCommentMarkers(raw).split("\n"));
  return {
    description: segments[0].trim(),
    tags: segments.slice(1).map(parseTag),
  };
}

function stripCommentMarkers(raw) {
  return raw
    .replace(/^\/\*\*\s*\n?/, "")
    .replace(/\s*\*\/$/, "")
    .split("\n")
    .map((line) => line.replace(/^\s*\* ?/, ""))
    .join("\n")
    .trimEnd();
}

// First segment is the free-form description; each subsequent segment starts
// with `@tag` at column 0. Tags inside fenced code stay with their parent.
function splitIntoSegments(lines) {
  const segments = [""];
  let inFence = false;
  for (const line of lines) {
    if (FENCE.test(line.trim())) inFence = !inFence;
    if (!inFence && TAG_START.test(line)) {
      segments.push(line);
    } else {
      const i = segments.length - 1;
      segments[i] += segments[i] ? "\n" + line : line;
    }
  }
  return segments;
}

function parseTag(raw) {
  const nameMatch = raw.match(/^@([A-Za-z]+)/);
  if (!nameMatch) return { tag: "unknown", raw };
  const tag = nameMatch[1];
  let rest = raw.slice(nameMatch[0].length).replace(/^[ \t]+/, "");

  // `@default` is the only tag whose payload starts with a brace but is *not*
  // a JSDoc type expression — skip the type extraction for it.
  let type = null;
  if (rest.startsWith("{") && tag !== "default") {
    const extracted = extractBalancedBraces(rest, 0);
    if (extracted) {
      type = extracted.content.trim();
      rest = rest.slice(extracted.endIndex).replace(/^[ \t]+/, "");
    }
  }

  switch (tag) {
    case "param":
    case "property":
    case "prop":
      return { tag, type, ...parseNameAndRest(rest) };
    case "return":
    case "returns":
      return { tag: "returns", type, description: trimDash(rest) };
    case "throws":
      return { tag, type, description: trimDash(rest) };
    case "typedef": {
      const m = rest.match(new RegExp(`^(${IDENT_PAT})\\s*([\\s\\S]*)$`));
      return { tag, type, name: m?.[1] ?? null, description: m?.[2].trim() ?? "" };
    }
    case "callback":
      return { tag, name: rest.trim().split(/\s+/)[0] || null };
    case "template": {
      // `@template {Constraint} Name description` — just the first identifier
      // after any optional constraint is the template-parameter name.
      const m = rest.match(/^([A-Za-z_$][\w$]*)/);
      return { tag, type, name: m?.[1] ?? null };
    }
    case "type":
      return { tag, type };
    case "default":
      return { tag, value: rest.trim() };
    case "module":
      return { tag, name: rest.trim() };
    case "file":
    case "fileoverview":
      return { tag: "file", description: rest.trim() };
    case "see":
      return { tag, description: rest.trim() };
    case "example":
      return { tag, body: rest };
    case "skillExample":
      return { tag, task: rest.trim() };
    case "skillCategory":
      return { tag, slug: rest.trim() };
    case "doctest":
      return { tag };
    default:
      return { tag, type, description: rest.trim() };
  }
}

// Parse `[name=default] description` or `name description`.
function parseNameAndRest(rest) {
  if (rest.startsWith("[")) {
    const close = findMatchingBracket(rest, 0);
    if (close !== -1) {
      const inner = rest.slice(1, close);
      const eq = inner.indexOf("=");
      return {
        name: eq >= 0 ? inner.slice(0, eq).trim() : inner.trim(),
        defaultValue: eq >= 0 ? inner.slice(eq + 1).trim() : null,
        optional: true,
        description: trimDash(rest.slice(close + 1)),
      };
    }
  }
  const m = rest.match(new RegExp(`^(${DOTTED_PAT})\\s*([\\s\\S]*)$`));
  return {
    name: m?.[1] ?? null,
    defaultValue: null,
    optional: false,
    description: trimDash(m?.[2] ?? rest),
  };
}

export function extractBalancedBraces(text, start) {
  if (text[start] !== "{") return null;
  let depth = 1;
  let i = start + 1;
  while (i < text.length && depth > 0) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}") depth--;
    i++;
  }
  return depth === 0 ? { content: text.slice(start + 1, i - 1), endIndex: i } : null;
}

function findMatchingBracket(text, start) {
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === "[") depth++;
    else if (text[i] === "]" && --depth === 0) return i;
  }
  return -1;
}

function trimDash(text) {
  return text.replace(/^[ \t]*-?\s*/, "").trim();
}
