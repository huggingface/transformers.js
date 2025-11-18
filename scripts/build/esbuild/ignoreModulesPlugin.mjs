/**
 * Plugin to ignore/exclude certain modules by returning an empty module.
 * Equivalent to webpack's resolve.alias with false value.
 */
export const ignoreModulesPlugin = (modules = []) => ({
  name: "ignore-modules",
  setup(build) {
    const filter = new RegExp(`^(${modules.join("|")})$`);
    build.onResolve({ filter }, (args) => {
      return { path: args.path, namespace: "ignore-modules" };
    });
    build.onLoad({ filter: /.*/, namespace: "ignore-modules" }, () => {
      return { contents: "export default {}" };
    });
  },
});