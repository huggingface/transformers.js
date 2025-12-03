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
    build.onLoad({ filter: /.*/, namespace: "ignore-modules" }, () => {
      return {
        contents: `
          const noop = () => {};
          const emptyObj = {};
          export default emptyObj;
          export const Readable = { fromWeb: noop };
          export const pipeline = noop;
          export const createWriteStream = noop;
          export const createReadStream = noop;
        `,
      };
    });
  },
});
