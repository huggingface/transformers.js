import path from "node:path";
import { fileURLToPath } from "node:url";

export const DIST_FOLDER = "dist";
export const EXTERNAL_MODULES = ["@huggingface/transformers"];

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT_DIR = path.join(__dirname, "../..");
export const OUT_DIR = path.join(ROOT_DIR, DIST_FOLDER);

export const getEsbuildDevConfig = (rootDir) => ({
  bundle: true,
  treeShaking: true,
  logLevel: "info",
  entryPoints: [path.join(rootDir, "src/index.ts")],
  platform: "browser",
  format: "esm",
  sourcemap: true,
});

export const getEsbuildProdConfig = (rootDir) => ({
  ...getEsbuildDevConfig(rootDir),
  logLevel: "warning",
  sourcemap: false,
});
