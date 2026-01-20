/**
 * Build target configuration
 * Each target defines a specific build variant
 */
export const BUILD_TARGETS = [
  {
    name: "ESM Build",
    config: {
      outputFile: "index.js",
      platform: "neutral",
      minify: false,
      includeInDevMode: true,
    },
  },
  {
    name: "Browser Build",
    config: {
      outputFile: "browser.js",
      platform: "browser",
      minify: false,
      includeInDevMode: false,
    },
  },
  {
    name: "Minified Browser Build",
    config: {
      outputFile: "browser.min.js",
      platform: "browser",
      minify: true,
      includeInDevMode: false,
    },
  },
];
