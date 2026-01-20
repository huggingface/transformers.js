import { colors } from "./logger.mjs";

/**
 * Plugin to log rebuild information in watch mode
 * @param {string} targetName - Name of the build target
 * @param {Object} log - Logger instance
 * @param {Object} options - Plugin options
 * @param {boolean} options.logFirstBuild - Whether to log the first build completion (default: true)
 */
export function rebuildPlugin(targetName, log, options = {}) {
  const { logFirstBuild = true } = options;

  return {
    name: "rebuild-logger",
    setup(build) {
      let startTime = 0;
      let isFirstBuild = true;

      build.onStart(() => {
        startTime = performance.now();
        if (!isFirstBuild) {
          log.build(`${colors.gray}Rebuilding ${targetName}...${colors.reset}`);
        }
      });

      build.onEnd((result) => {
        const endTime = performance.now();
        const duration = (endTime - startTime).toFixed(2);

        if (result.errors.length > 0) {
          log.error(
            `${colors.bright}${targetName}${colors.reset} - Build failed with ${result.errors.length} error(s) in ${duration}ms`,
          );
        } else if (!isFirstBuild) {
          log.done(
            `${colors.bright}${targetName}${colors.reset} - Rebuilt in ${colors.gray}${duration}ms${colors.reset}`,
          );
        } else if (logFirstBuild) {
          log.done(
            `${colors.bright}${targetName}${colors.reset} - Built in ${colors.gray}${duration}ms${colors.reset}`,
          );
        }

        isFirstBuild = false;
      });
    },
  };
}
