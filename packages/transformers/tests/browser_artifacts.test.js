import { build } from "esbuild";
import { fileURLToPath } from "node:url";

// These are consumption tests of freshly built distribution files, not source-import scans.
// Run the package build before this suite. No Node polyfills, aliases or externalization are used.
const artifacts = ["transformers.web.js", "transformers.web.min.js", "transformers.js", "transformers.min.js"];
const dist = fileURLToPath(new URL("../dist/", import.meta.url));
const outdir = dist + "browser-consumption";
const entry = (name) => dist + name;
const inspect = async (name) => {
  const result = await build({ entryPoints: [entry(name)], bundle: false, write: false, metafile: true, logLevel: "silent", outdir });
  // With bundle:false, dependency records live on the emitted module, not the unscanned input.
  return Object.values(result.metafile.outputs).flatMap((output) => output.imports);
};

describe("browser distribution consumption without Node polyfills", () => {
  it.each(artifacts)("does not import Node timers from %s", async (name) => {
    const imports = await inspect(name);
    expect(imports.filter(({ path }) => /^(?:node:)?timers\/promises$/.test(path))).toEqual([]);
    if (name.startsWith("transformers.web.")) expect(imports.some(({ path }) => path.startsWith("onnxruntime-web"))).toBe(true);
  });

  it.each(artifacts)(
    "rebundles %s for platform:browser",
    async (name) => {
      const result = await build({
        entryPoints: [entry(name)],
        bundle: true,
        platform: "browser",
        format: "esm",
        write: false,
        metafile: true,
        logLevel: "silent",
        outdir,
      });
      expect(result.outputFiles.length).toBeGreaterThan(0);
      expect(
        Object.values(result.metafile.outputs)
          .flatMap((output) => output.imports)
          .filter(({ external }) => external),
      ).toEqual([]);
    },
    30000,
  );

  it.each(["transformers.js", "transformers.min.js"])("keeps %s self-contained at its static import boundary", async (name) => {
    expect((await inspect(name)).filter(({ kind }) => kind === "import-statement")).toEqual([]);
  });
});
