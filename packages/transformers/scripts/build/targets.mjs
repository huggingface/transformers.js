import { NODE_IGNORE_MODULES, NODE_EXTERNAL_MODULES, WEB_IGNORE_MODULES, WEB_EXTERNAL_MODULES, REACT_NATIVE_IGNORE_MODULES, REACT_NATIVE_EXTERNAL_MODULES } from "./constants.mjs";

/**
 * Build target configuration
 * Each target defines a specific build variant
 */
export const BUILD_TARGETS = [
  {
    name: "Bundle Build (ESM)",
    config: {
      name: "",
      suffix: ".js",
      format: "esm",
      ignoreModules: WEB_IGNORE_MODULES,
      usePostBuild: true,
    },
  },
  {
    name: "Web Build (ESM)",
    config: {
      name: ".web",
      suffix: ".js",
      format: "esm",
      ignoreModules: WEB_IGNORE_MODULES,
      externalModules: WEB_EXTERNAL_MODULES,
      usePostBuild: false,
    },
  },
  {
    name: "React Native Build (ESM)",
    config: {
      name: ".native",
      suffix: ".mjs",
      format: "esm",
      ignoreModules: REACT_NATIVE_IGNORE_MODULES,
      externalModules: REACT_NATIVE_EXTERNAL_MODULES,
      usePostBuild: false,
    },
  },
  {
    name: "React Native Build (CJS)",
    config: {
      name: ".native",
      suffix: ".cjs",
      format: "cjs",
      ignoreModules: REACT_NATIVE_IGNORE_MODULES,
      externalModules: REACT_NATIVE_EXTERNAL_MODULES,
      usePostBuild: false,
    },
  },
  {
    name: "Node Build (ESM)",
    config: {
      name: ".node",
      suffix: ".mjs",
      format: "esm",
      ignoreModules: NODE_IGNORE_MODULES,
      externalModules: NODE_EXTERNAL_MODULES,
      usePostBuild: false,
    },
  },
  {
    name: "Node Build (CJS)",
    config: {
      name: ".node",
      suffix: ".cjs",
      format: "cjs",
      ignoreModules: NODE_IGNORE_MODULES,
      externalModules: NODE_EXTERNAL_MODULES,
      usePostBuild: false,
    },
  },
];
