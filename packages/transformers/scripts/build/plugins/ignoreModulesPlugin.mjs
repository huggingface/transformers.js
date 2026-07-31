/**
 * Plugin to ignore/exclude certain modules by returning an empty module.
 * Equivalent to webpack's resolve.alias with false value.
 */
export const ignoreModulesPlugin = (modules = []) => ({
  name: "ignore-modules",
  setup(build) {
    // Escape special regex characters in module names
    const escapeRegex = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const escapedModules = modules.map(escapeRegex);

    // Match both "module" and "node:module" patterns
    const patterns = escapedModules.flatMap((mod) => [mod, `node:${mod}`]);
    const filter = new RegExp(`^(${patterns.join("|")})$`);

    build.onResolve({ filter }, (args) => {
      return { path: args.path, namespace: "ignore-modules" };
    });
    build.onLoad({ filter: /.*/, namespace: "ignore-modules" }, (args) => {
      switch (args.path) {
        case "node:stream":
          return {
            contents: `
              const noop = () => {};
              export default {};
              export const Readable = { fromWeb: noop };
            `,
          };
        case "node:stream/promises":
          return {
            contents: `
              const noop = () => {};
              export default {};
              export const pipeline = noop;
            `,
          };
        case "node:fs":
        case "node:path":
        case "node:url":
        case "sharp":
        case "onnxruntime-node":
        default:
          return {
            contents: `export default {};`,
          };
      }
    });
  },
});
