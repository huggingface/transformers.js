/**
 * Transform imports in built files to use URLs
 * Replaces bare module specifiers with CDN/dev server URLs
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { colors, createLogger } from "../../../scripts/logger.mjs";

const log = createLogger("react");
const TRANSFORMERS_PACKAGE = "@huggingface/transformers";
const DEV_SERVER_URL = "http://localhost:8080/transformers.js";

function transformImports(filePath) {
  const content = readFileSync(filePath, "utf-8");
  let transformed = content;
  let changeCount = 0;

  // Transform @huggingface/transformers imports
  const transformersRegex = new RegExp(
    `import\\s+([^;]+?)\\s+from\\s+["']${TRANSFORMERS_PACKAGE}["'];`,
    "g",
  );

  transformed = transformed.replace(transformersRegex, (match, imports) => {
    changeCount++;
    return `import ${imports} from "${DEV_SERVER_URL}";`;
  });

  if (transformed !== content) {
    writeFileSync(filePath, transformed, "utf-8");
    return changeCount;
  }

  return 0;
}

function processDirectory(dir) {
  let transformCount = 0;
  let fileCount = 0;

  const entries = readdirSync(dir);

  for (const entry of entries) {
    const fullPath = path.join(dir, entry);
    const stat = statSync(fullPath);

    if (stat.isDirectory()) {
      const result = processDirectory(fullPath);
      transformCount += result;
    } else if (entry.endsWith(".js") || entry.endsWith(".mjs")) {
      const changes = transformImports(fullPath);
      if (changes > 0) {
        fileCount++;
        const relativePath = path.relative(process.cwd(), fullPath);
        log.dim(
          `  ${relativePath}: ${changes} import${changes !== 1 ? "s" : ""} transformed`,
        );
        transformCount += changes;
      }
    }
  }

  return transformCount;
}

export function transformAllImports(distDir) {
  log.info("Transforming imports to use CDN/dev server URLs...");
  const count = processDirectory(distDir);

  if (count > 0) {
    log.success(
      `Transformed ${count} import${count !== 1 ? "s" : ""} to use URLs`,
    );
  } else {
    log.dim("No imports to transform");
  }
}
