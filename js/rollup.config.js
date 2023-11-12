import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';

export default {
  input: 'js/index.js',
  external: ['d3'],
  output: {
    file: 'build/vexpr.js',
    globals: {d3: 'd3'},
    format: 'umd',
    extend: true,
    name: 'vexpr' // The global variable name to use in the browser
  },
  plugins: [resolve(), commonjs()]
};
