import fs from "node:fs";
import path from "node:path";

import { listFiles } from "./fs.mjs";
import { apiOutputDir } from "./paths.mjs";
import { buildLinkIndexes, hasRenderableContent, renderModule } from "./render-api.mjs";

export function generateApiDocs({ project }) {
  clearExistingMarkdown();

  const errors = [];
  const linkIndexes = buildLinkIndexes(project.ir, project.publicNames);

  for (const mod of project.ir.modules) {
    if (!hasRenderableContent(mod, project.publicNames)) {
      console.log(`skipped ${mod.name}.md — no public content`);
      continue;
    }

    try {
      const rendered = renderModule(mod, project.ir, { publicNames: project.publicNames, linkIndexes });
      const outputPath = path.resolve(apiOutputDir, `${mod.name}.md`);
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      fs.writeFileSync(outputPath, rendered);
      console.log(`wrote ${mod.name}.md`);
    } catch (err) {
      errors.push(`${mod.name}.md: ${err.message}`);
    }
  }

  return { errors };
}

function clearExistingMarkdown() {
  for (const file of listFiles(apiOutputDir, ".md")) fs.unlinkSync(file);
}
