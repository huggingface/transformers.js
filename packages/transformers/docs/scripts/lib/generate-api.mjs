import fs from "node:fs";
import path from "node:path";

import { apiOutputDir, packageRoot } from "./paths.mjs";
import { loadProject } from "./load.mjs";
import { renderModule } from "./render-api.mjs";

export function generateApiDocs({ project = loadProject(packageRoot), outputDir = apiOutputDir, log = console.log } = {}) {
  clearExistingMarkdown(outputDir);

  const written = [];
  const skipped = [];

  for (const mod of project.ir.modules) {
    const rendered = renderModule(mod, project.ir, { publicNames: project.publicNames });
    if (!hasPublicBody(rendered)) {
      skipped.push(mod.name);
      log(`skipped ${mod.name}.md — no public content`);
      continue;
    }

    const outputPath = path.resolve(outputDir, `${mod.name}.md`);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, rendered);
    written.push(path.relative(outputDir, outputPath));
    log(`wrote ${mod.name}.md`);
  }

  return { written, skipped };
}

function hasPublicBody(markdown) {
  if (/^## /m.test(markdown)) return true;
  const body = markdown.replace(/^# [^\n]+\n/, "");
  return /\]\(/.test(body);
}

function clearExistingMarkdown(outputDir) {
  if (!fs.existsSync(outputDir)) return;
  for (const entry of fs.readdirSync(outputDir, { recursive: true })) {
    if (entry.endsWith(".md")) fs.unlinkSync(path.join(outputDir, entry));
  }
}
