/**
 * @file Entry point for the Transformers.js Web Worker utilities.
 * Provides helper functions for using Transformers.js pipelines in Web Workers.
 *
 * @module transformers-webworker
 */

// Web Worker utilities
export { default as webWorkerPipeline } from './webWorkerPipeline';
export type { WebWorkerPipeline } from './webWorkerPipeline';
export { default as webWorkerPipelineHandler } from './webWorkerPipelineHandler';
