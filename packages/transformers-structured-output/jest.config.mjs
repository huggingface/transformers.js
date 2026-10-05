/** @type {import('jest').Config} */
export default {
  clearMocks: true,
  collectCoverage: true,
  coverageDirectory: "coverage",
  coveragePathIgnorePatterns: ["node_modules", "tests"],
  coverageProvider: "v8",
  roots: ["./tests/"],
  testTimeout: 32000,
  transform: {},
};
