import { EXTERNAL_MODULES } from "./constants.mjs";

/**
 * Build target configuration
 * Each target defines a specific build variant
 */
export const BUILD_TARGETS = [
  {
    name: "Web Worker Build (ESM)",
    config: {
      name: "",
      suffix: ".js",
      format: "esm",
      externalModules: EXTERNAL_MODULES,
    },
  },
];
