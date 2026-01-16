import { colors, createLogger } from "../../../../../../scripts/logger.mjs";

const log = createLogger("transformers");

/**
 * Plugin to log rebuild events with timing
 */
export const rebuildPlugin = (name) => {
  let startTime = 0;
  let isFirstBuild = true;

  return {
    name: "rebuild-logger",
    setup(build) {
      build.onStart(() => {
        startTime = performance.now();
        if (!isFirstBuild) {
          log.build(`${colors.gray}Rebuilding ${name}...${colors.reset}`);
        }
      });

      build.onEnd((result) => {
        const endTime = performance.now();
        const duration = (endTime - startTime).toFixed(2);

        if (result.errors.length > 0) {
          log.error(
            `${colors.bright}${name}${colors.reset} - Build failed with ${result.errors.length} error(s) in ${duration}ms`,
          );
        } else if (!isFirstBuild) {
          log.done(
            `${colors.bright}${name}${colors.reset} - Rebuilt in ${colors.gray}${duration}ms${colors.reset}`,
          );
        } else {
          log.done(
            `${colors.bright}${name}${colors.reset} - Built in ${colors.gray}${duration}ms${colors.reset}`,
          );
        }

        isFirstBuild = false;
      });
    },
  };
};
