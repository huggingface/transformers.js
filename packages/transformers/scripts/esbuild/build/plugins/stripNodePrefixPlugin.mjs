/**
 * Plugin to strip the "node:" prefix from module requests.
 * Equivalent to webpack's StripNodePrefixPlugin.
 */
export const stripNodePrefixPlugin = () => ({
  name: "strip-node-prefix",
  setup(build) {
    build.onResolve({ filter: /^node:/ }, (args) => {
      return {
        path: args.path.replace(/^node:/, ""),
        external: true,
      };
    });
  },
});
