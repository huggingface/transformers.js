/**
 * Plugin to log rebuild events with timing
 */
export const rebuildPlugin = (name) => {
  let startTime = 0;

  return {
    name: "rebuild-logger",
    setup(build) {
      build.onStart(() => {
        startTime = performance.now();
      });

      build.onEnd((result) => {
        const endTime = performance.now();
        const duration = (endTime - startTime).toFixed(2);

        if (result.errors.length > 0) {
          console.log(`\n${name} - Build failed with ${result.errors.length} error(s) in ${duration}ms`);
        } else {
          console.log(`\n${name} - Rebuilt in ${duration}ms`);
        }
      });
    },
  };
};
