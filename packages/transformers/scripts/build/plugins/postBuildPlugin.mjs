import path from "node:path";
import { copyFileSync, unlinkSync, existsSync } from "node:fs";
import { colors, createLogger } from "../../../../../scripts/logger.mjs";

const log = createLogger("transformers");

/**
 * Plugin to post-process build files.
 * Equivalent to webpack's PostBuildPlugin.
 */
export const postBuildPlugin = (distDir, rootDir) => {
  // it should copy the files only once. In watch mode for example it should not rerun every time
  let completed = false;

  return {
    name: "post-build",
    setup(build) {
      build.onEnd(() => {
        if (completed) return;
        completed = true;

        const ORT_JSEP_FILE = "ort-wasm-simd-threaded.jsep.mjs";
        const ORT_BUNDLE_FILE = "ort.bundle.min.mjs";

        // 1. Remove unnecessary files
        const file = path.join(distDir, ORT_BUNDLE_FILE);
        if (existsSync(file)) unlinkSync(file);

        // 2. Copy unbundled JSEP file
        const ORT_SOURCE_DIR = path.join(rootDir, "node_modules/onnxruntime-web/dist");
        const src = path.join(ORT_SOURCE_DIR, ORT_JSEP_FILE);

        if (existsSync(src)) {
          const dest = path.join(distDir, ORT_JSEP_FILE);
          copyFileSync(src, dest);
          log.success(`${colors.gray}Copied ${ORT_JSEP_FILE}${colors.reset}`);
        } else {
          log.warning(`Could not find ${ORT_JSEP_FILE} in node_modules`);
        }
      });
    },
  };
};
