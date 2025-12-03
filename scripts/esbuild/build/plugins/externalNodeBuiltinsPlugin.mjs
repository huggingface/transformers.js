/**
 * Plugin to automatically mark all node:* imports as external.
 * This prevents having to manually list all Node.js built-in modules.
 */
export const externalNodeBuiltinsPlugin = () => ({
  name: "external-node-builtins",
  setup(build) {
    // Mark all node:* imports as external
    build.onResolve({ filter: /^node:/ }, (args) => ({
      path: args.path,
      external: true,
    }));
  },
});
