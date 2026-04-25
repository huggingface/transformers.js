import path from "node:path";

import { packageRoot, repoRoot, skillDir as defaultSkillDir } from "./paths.mjs";
import { loadProject } from "./load.mjs";
import { renderSkill } from "./render-skill.mjs";

export function generateSkillDocs({ project = loadProject(packageRoot), skillDir = defaultSkillDir, log = console.log } = {}) {
  renderSkill({
    ir: project.ir,
    tasks: project.tasks,
    publicNames: project.publicNames,
    skillDir,
  });

  log(`wrote skill to ${path.relative(repoRoot, skillDir) || skillDir}`);
  return { skillDir };
}
