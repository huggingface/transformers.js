import fs from "node:fs";
import path from "node:path";

import { apiOutputDir, toctreePath } from "./paths.mjs";

export function validateGeneratedDocs({ outputDir = apiOutputDir, tocPath = toctreePath } = {}) {
  const generatedApiPages = listApiPages(outputDir);
  const linkedApiPages = readToctreeApiPages(tocPath);
  const sourceDir = path.dirname(tocPath);
  const brokenLinks = validateInternalLinks(sourceDir);

  const unlisted = difference(generatedApiPages, linkedApiPages);
  const stale = difference(linkedApiPages, generatedApiPages);

  return {
    ok: unlisted.length === 0 && stale.length === 0 && brokenLinks.length === 0,
    unlisted,
    stale,
    brokenLinks,
  };
}

export function formatValidationResult(result) {
  if (result.ok) return "validated docs toctree";

  const lines = ["docs validation failed"];
  if (result.unlisted.length) {
    lines.push("", "Generated API pages missing from _toctree.yml:");
    for (const page of result.unlisted) lines.push(`- ${page}`);
  }
  if (result.stale.length) {
    lines.push("", "API pages listed in _toctree.yml but not generated:");
    for (const page of result.stale) lines.push(`- ${page}`);
  }
  if (result.brokenLinks.length) {
    lines.push("", "Broken local markdown links:");
    for (const link of result.brokenLinks) {
      const anchor = link.anchor ? `#${link.anchor}` : "";
      lines.push(`- ${link.file}: ${link.target} -> ${link.resolved}${anchor} (${link.type})`);
    }
  }
  return lines.join("\n");
}

function listApiPages(outputDir) {
  return listMarkdown(outputDir)
    .map((file) => toApiPage(outputDir, file))
    .sort();
}

function listMarkdown(dir) {
  if (!fs.existsSync(dir)) return [];

  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listMarkdown(full));
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

function toApiPage(outputDir, file) {
  const relative = path.relative(outputDir, file).replaceAll(path.sep, "/").replace(/\.md$/, "");
  return `api/${relative}`;
}

function readToctreeApiPages(tocPath) {
  if (!fs.existsSync(tocPath)) return [];

  const pages = [];
  const re = /^\s*-\s+local:\s+(api\/\S+)\s*$/gm;
  const text = fs.readFileSync(tocPath, "utf8");
  for (const match of text.matchAll(re)) pages.push(match[1]);
  return pages.sort();
}

function difference(a, b) {
  const right = new Set(b);
  return a.filter((item) => !right.has(item));
}

function validateInternalLinks(sourceDir) {
  const files = listMarkdown(sourceDir);
  const fileSet = new Set(files.map((file) => relativeMarkdownPath(sourceDir, file)));
  const anchorsByFile = new Map(files.map((file) => [relativeMarkdownPath(sourceDir, file), collectAnchors(file)]));
  const issues = [];

  for (const file of files) {
    const rel = relativeMarkdownPath(sourceDir, file);
    const text = readMarkdownWithIncludes(file);

    for (const match of text.matchAll(MARKDOWN_LINK_RE)) {
      const target = match[1] ?? match[2];
      if (!isLocalLink(target)) continue;

      const [targetPath, rawAnchor = ""] = target.split("#");
      let resolved = rel;
      if (targetPath) {
        resolved = resolveMarkdownTarget(rel, targetPath);
        if (!fileSet.has(resolved)) {
          issues.push({ type: "missing-file", file: rel, target, resolved });
          continue;
        }
      }

      if (rawAnchor) {
        const anchor = decodeURIComponent(rawAnchor);
        if (!anchorsByFile.get(resolved)?.has(anchor)) {
          issues.push({ type: "missing-anchor", file: rel, target, resolved, anchor });
        }
      }
    }
  }

  return issues;
}

const MARKDOWN_LINK_RE = /!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;

function isLocalLink(target) {
  return !!target && !/^[a-z][a-z0-9+.-]*:/i.test(target) && !target.startsWith("//");
}

function resolveMarkdownTarget(fromFile, target) {
  let resolved = path.posix.normalize(path.posix.join(path.posix.dirname(fromFile), target));
  if (!path.posix.extname(resolved)) resolved += ".md";
  return resolved;
}

function relativeMarkdownPath(sourceDir, file) {
  return path.relative(sourceDir, file).replaceAll(path.sep, "/");
}

function collectAnchors(file) {
  const text = readMarkdownWithIncludes(file);
  const anchors = new Set([""]);

  for (const match of text.matchAll(/<a\s+id=["']([^"']+)["']/g)) anchors.add(match[1]);
  for (const match of text.matchAll(/^#{1,6}\s+(.+)$/gm)) anchors.add(slugHeading(match[1]));

  return anchors;
}

function readMarkdownWithIncludes(file, seen = new Set()) {
  if (seen.has(file)) return "";
  seen.add(file);

  const text = fs.readFileSync(file, "utf8");
  return text.replace(/<include>\s*([\s\S]*?)\s*<\/include>/g, (match, rawConfig) => {
    const includePath = rawConfig.match(/"path"\s*:\s*"([^"]+)"/)?.[1];
    if (!includePath) return match;

    const resolved = path.resolve(path.dirname(file), includePath);
    return fs.existsSync(resolved) ? readMarkdownWithIncludes(resolved, seen) : match;
  });
}

function slugHeading(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/<[^>]+>/g, "")
    .replace(/[`*_~]/g, "")
    .replace(/&[a-z]+;/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}
