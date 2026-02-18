/**
 * JSDoc plugin to transform TypeScript-style type expressions into JSDoc-compatible syntax.
 */

function extractBalancedBraces(text, start) {
  if (text[start] !== "{") return null;
  let depth = 1,
    i = start + 1;
  while (i < text.length && depth > 0) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}") depth--;
    i++;
  }
  return depth === 0 ? { content: text.slice(start + 1, i - 1), endIndex: i } : null;
}

function transformType(expr) {
  let result = expr,
    prev = "";
  while (result !== prev) {
    prev = result;
    result = result
      .replace(/import\([^)]+\)((?:\.\w+)*)/g, (_, p) => p?.slice(1) || "any") // import() -> Type
      .replace(/\w+\[['"][^\]]+['"]\]\s+extends\s+[^}]+/g, "any") // X['p'] extends ... -> any
      .replace(/(\w+)(?:<[^>]+>)?\[[^\]]+\]/g, "$1") // Type[x] or Type<T>[x] -> Type
      .replace(/keyof\s+typeof\s+\w+/g, "string") // keyof typeof X -> string
      .replace(/typeof\s+\w+/g, "Object") // typeof X -> Object
      .replace(/(\w+)\?\s*:/g, "$1:") // x?: T -> x: T
      .replace(/;\s*([}\n])/g, " $1")
      .replace(/;\s+/g, ", ") // semicolons -> commas
      .replace(/\{\s*\[\w+\s+in\s+[^\]]+\][^}]*\}/g, "Object") // mapped types -> Object
      .replace(/new\s*\([^)]*\)\s*=>\s*\{[^}]*\}/g, "Function") // new () => {...} -> Function
      .replace(/new\s*\([^)]*\)\s*=>\s*\w+/g, "Function") // new () => T -> Function
      .replace(/\([^()]*\)\s*=>\s*\w[\w<>|[\]]*/g, "Function") // () => T -> Function
      .replace(/\{[^{}]*\}\[\]/g, "Array") // {x:T}[] -> Array
      .replace(/(\w+)<[^>]+>\[\]/g, "Array.<$1>") // T<U>[] -> Array.<T>
      .replace(/\([^()]+\)\[\]/g, "Array") // (A|B)[] -> Array
      .replace(/(\w+)\[\]/g, "Array") // T[] -> Array (simple)
      .replace(/\[\w+\]/g, "Array") // [T] single-element tuple -> Array
      .replace(/\[[^\[\]]*,[^\[\]]*\]/g, "Array") // tuples with commas -> Array
      .replace(/\w+\s+extends\s+[^?]+\?\s*[^:]+\s*:\s*[^,}>)]+/g, "any") // conditionals -> any
      .replace(/\bnew\s+([A-Z]\w*)\b/g, "$1"); // new Type -> Type
    if (!result.includes("=>")) result = result.replace(/\s*&\s*/g, "|"); // A & B -> A|B
  }
  return result;
}

function transformComment(comment) {
  const tagRegex =
    /@(?:type|param|returns?|typedef|property|prop|callback|template|augments|extends|class|constructor|member|var|const|enum|throws?)[^\S\n]*\{/g;
  let result = "",
    lastIndex = 0,
    match;
  while ((match = tagRegex.exec(comment)) !== null) {
    const braceStart = match.index + match[0].length - 1;
    result += comment.slice(lastIndex, braceStart);
    const extracted = extractBalancedBraces(comment, braceStart);
    if (extracted) {
      result += "{" + transformType(extracted.content) + "}";
      lastIndex = tagRegex.lastIndex = extracted.endIndex;
    } else {
      result += "{";
      lastIndex = braceStart + 1;
    }
  }
  return result + comment.slice(lastIndex);
}

export const handlers = {
  beforeParse: (e) => {
    e.source = e.source
      .replace(/\/\*\*[\s\S]*?\*\//g, transformComment) // transform JSDoc comments
      .replace(/\b(\d+)n\b/g, "BigInt($1)"); // BigInt literals
  },
};
