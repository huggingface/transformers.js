import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT_DIR = path.join(__dirname, "../../..");
export const OUT_DIR = path.join(ROOT_DIR, "dist");
export const SRC_DIR = path.join(ROOT_DIR, "src");

export const EXTERNAL_MODULES = ["react", "@huggingface/transformers"];

export const getEsbuildDevConfig = () => ({
  entryPoints: [path.join(SRC_DIR, "index.ts")],
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2020",
  sourcemap: true,
  external: EXTERNAL_MODULES,
  logLevel: "silent",
});

export const getEsbuildProdConfig = () => ({
  ...getEsbuildDevConfig(),
  logLevel: "warning",
  sourcemap: true,
});
