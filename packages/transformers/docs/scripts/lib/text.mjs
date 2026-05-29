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
// `[`TypeName`]` (reST cross-reference syntax). Drop the leading artifact.
export function stripDocArtifacts(text) {
  return text.replace(/^\[`?[A-Za-z_$][\w$.]*`?\]\s*/, "");
}
