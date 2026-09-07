// Bracket- and string-aware text scanners shared between IR-building and
// markdown rendering. Type strings can contain nested generics, tuples,
// function-type arrow lists, etc.; naive split/find on `,`/`|`/`&` would
// shred them. These helpers know to skip separators that appear inside
// `<>`, `()`, `{}`, `[]`, or quoted strings.

const OPEN = "<({[";
const CLOSE = ">)}]";

// The `>` of a function-type arrow (`=>`) is not a closing angle bracket.
function isArrowTail(text, i) {
  return text[i] === ">" && text[i - 1] === "=";
}

// Return the index of the first occurrence of `needle` at top level (depth
// zero), or -1 if it doesn't appear outside brackets/strings.
export function findTopLevel(text, needle) {
  let depth = 0;
  let inStr = null;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === inStr && text[i - 1] !== "\\") inStr = null;
    } else if (ch === '"' || ch === "'" || ch === "`") {
      inStr = ch;
    } else if (OPEN.includes(ch)) {
      depth++;
    } else if (CLOSE.includes(ch) && !isArrowTail(text, i)) {
      depth--;
    } else if (depth === 0 && ch === needle) {
      return i;
    }
  }
  return -1;
}

// Given `text[start] === open`, return the index of the matching `close`,
// skipping nested pairs and quoted strings; -1 when absent or unbalanced.
export function matchingBracket(text, start, open, close) {
  if (text[start] !== open) return -1;
  let depth = 1;
  let inStr = null;
  for (let i = start + 1; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === inStr && text[i - 1] !== "\\") inStr = null;
    } else if (ch === '"' || ch === "'" || ch === "`") {
      inStr = ch;
    } else if (ch === open) {
      depth++;
    } else if (ch === close && !isArrowTail(text, i) && --depth === 0) {
      return i;
    }
  }
  return -1;
}

// Split `text` on `sep`, ignoring separators inside brackets/strings.
// Returns `[text]` (the unsplit input) when no split occurred — callers
// can use that to detect "this was a single chunk all along".
export function splitTopLevel(text, sep) {
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
    } else if (OPEN.includes(ch)) {
      depth++;
      buf += ch;
    } else if (CLOSE.includes(ch) && !isArrowTail(text, i)) {
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
