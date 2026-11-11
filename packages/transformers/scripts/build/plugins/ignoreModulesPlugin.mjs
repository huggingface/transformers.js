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

    // Match "module", "node:module", and any subpath export of them (e.g. "onnxruntime-web/webgpu").
    // Without the subpath group, a package listed here would still be bundled when imported via a subpath.
    const filter = new RegExp(`^(node:)?(${escapedModules.join("|")})(/.*)?$`);

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
        case "buffer":
          return {
            contents: `export const Buffer = globalThis.Buffer; export default { Buffer: globalThis.Buffer };`,
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
