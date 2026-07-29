import { getOnnxProviderModule } from './default.js';

/**
 * Provider operations used by model-file discovery.
 *
 * @typedef {Object} ModelRegistryInferenceProvider
 * @property {(options: Object) => ReadonlyArray<string>|Promise<ReadonlyArray<string>>} listModelArtifacts
 * @property {(options: Object) => Promise<string[]>} [getAvailableDtypes]
 * @property {(files: string[], sessions: Record<string, string>) => string[]} [filterModelArtifacts]
 */

/**
 * Resolve an explicitly supplied registry provider or the lazy default provider.
 *
 * @param {ModelRegistryInferenceProvider|null} [provider]
 * @returns {Promise<ModelRegistryInferenceProvider>}
 */
export async function getModelRegistryInferenceProvider(provider = null) {
    if (typeof provider?.listModelArtifacts === 'function') return provider;
    const providerClass = /** @type {any} */ (provider?.constructor);
    if (typeof providerClass?.listModelArtifacts === 'function') return providerClass;
    const { OnnxInferenceProvider } = await getOnnxProviderModule();
    return OnnxInferenceProvider;
}
