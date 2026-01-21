/**
 * The pipeline function should correctly infer:
 *  1. The type of the pipeline, based on the task name.
 *  2. The output type of the pipeline, based on the types of the inputs.
 * 
 * To test this, we create pipelines for various tasks, and call them with different types of inputs.
 * We then check that the output types are as expected.
 * 
 * Note: These tests are not meant to be executed, but rather to be type-checked by TypeScript.
 */
import { pipeline } from '../../src/transformers.js';
