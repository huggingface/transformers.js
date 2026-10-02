/**
 * Plugin to rewrite module imports to alternative packages.
 * Handles both bare (e.g. "path") and node:-prefixed (e.g. "node:path") specifiers.
 */
export const aliasPlugin = (aliases = {}) => ({
  name: "alias",
  setup(build) {
    const entries = Object.entries(aliases);
    if (entries.length === 0) return;

    const escaped = entries.map(([k]) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const filter = new RegExp(`^(node:)?(${escaped.join("|")})$`);

    build.onResolve({ filter }, async (args) => {
      const bare = args.path.replace(/^node:/, "");
      const target = aliases[bare];
      if (!target) return null;
      const result = await build.resolve(target, {
        resolveDir: args.resolveDir,
        kind: args.kind,
      });
      return result;
    });
  },
});
