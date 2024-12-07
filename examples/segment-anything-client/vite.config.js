import { defineConfig } from 'vite';
export default defineConfig(env => {
  const config = {
    build: {
      target: 'esnext'
    }
  };

  // TODO: Add this back when .wasm files are served locally
  // if (env.mode === 'development') {
  //   // The .wasm files are not correctly served using Vite in development mode.
  //   // This is a workaround to exclude the onnxruntime-web package from Vite's optimization.
  //   // See also: https://github.com/vitejs/vite/issues/8427
  //   config.optimizeDeps = { exclude: ["onnxruntime-web"] };
  // }

  return config;
});
