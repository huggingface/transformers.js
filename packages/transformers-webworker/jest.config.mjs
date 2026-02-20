/*
 * For a detailed explanation regarding each configuration property, visit:
 * https://jestjs.io/docs/configuration
 */

export default {
  clearMocks: true,
  collectCoverage: true,
  coverageDirectory: "coverage",
  coveragePathIgnorePatterns: ["node_modules", "tests"],
  coverageProvider: "v8",
  roots: ["./tests/"],
  preset: "ts-jest/presets/default-esm",
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        useESM: true,
        tsconfig: {
          moduleResolution: "node",
          esModuleInterop: true,
          skipLibCheck: true,
          isolatedModules: true,
        },
      },
    ],
  },
  moduleNameMapper: {
    "^(\\.{1,2}/.*)\\.js$": "$1",
  },
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "mjs"],
  testEnvironment: "node",
  extensionsToTreatAsEsm: [".ts"],
};
