import fs from "node:fs";
import path from "node:path";

import { listFiles } from "./fs.mjs";
import { apiOutputDir, packageRoot } from "./paths.mjs";
import { loadProject } from "./load.mjs";
import { buildLinkIndexes, hasRenderableContent, renderModule } from "./render-api.mjs";

export function generateApiDocs({ project = loadProject(packageRoot), outputDir = apiOutputDir, log = console.log } = {}) {
  clearExistingMarkdown(outputDir);

  const written = [];
  const skipped = [];
  const errors = [];
  const linkIndexes = buildLinkIndexes(project.ir, project.publicNames);

  for (const mod of project.ir.modules) {
    if (!hasRenderableContent(mod, project.publicNames)) {
      skipped.push(mod.name);
      log(`skipped ${mod.name}.md — no public content`);
      continue;
    }

    try {
      const rendered = renderModule(mod, project.ir, { publicNames: project.publicNames, linkIndexes });
      const outputPath = path.resolve(outputDir, `${mod.name}.md`);
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      fs.writeFileSync(outputPath, rendered);
      written.push(path.relative(outputDir, outputPath));
      log(`wrote ${mod.name}.md`);
    } catch (err) {
      errors.push(`${mod.name}.md: ${err.message}`);
    }
  }

  return { written, skipped, errors };
}

function clearExistingMarkdown(outputDir) {
  for (const file of listFiles(outputDir, ".md")) fs.unlinkSync(file);
}
