import typescript from '@rollup/plugin-typescript';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import copy from 'rollup-plugin-copy';
import { emptyDir } from 'rollup-plugin-empty-dir';

export default {
  input: {
    background: 'src/background.ts',
    content: 'src/content.ts',
    popup: 'src/popup.ts',
    sidepanel: 'src/sidepanel.ts'
  },
  output: {
    dir: 'dist',
    format: 'esm',
    entryFileNames: '[name].js',
    manualChunks: undefined // Prevent code splitting for Chrome extension
  },
  plugins: [
    emptyDir(),
    nodeResolve({
      browser: true,
      preferBuiltins: false
    }),
    commonjs(),
    typescript({
      tsconfig: './tsconfig.json',
      sourceMap: false
    }),
    copy({
      targets: [
        {
          src: 'public/*',
          dest: 'dist'
        },
        {
          src: 'src/*.html',
          dest: 'dist'
        },
        {
          src: 'src/*.css',
          dest: 'dist'
        },
        {
          src: 'node_modules/@xenova/transformers/dist/*.wasm',
          dest: 'dist'
        }
      ]
    })
  ],
  external: [],
  watch: {
    include: 'src/**'
  }
};
