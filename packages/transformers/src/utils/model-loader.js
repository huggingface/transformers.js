import { getModelFile, MAX_EXTERNAL_DATA_CHUNKS } from './hub.js';
import { apis } from '../env.js';

/**
 * Resolves an `use_external_data_format` config value to the number of data chunks for a given file.
 * @param {import('./hub.js').ExternalData|Record<string, import('./hub.js').ExternalData>|null|undefined} config The external data format configuration.
 * @param {string} fullName The full ONNX file name (e.g., "model_quantized.onnx").
 * @param {string} fileName The base file name (e.g., "model").
 * @returns {number} The number of external data chunks (0 if none).
 */
export function resolveExternalDataFormat(config, fullName, fileName) {
    if (!config) return 0;
    if (typeof config === 'object' && config !== null) {
        if (config.hasOwnProperty(fullName)) return +config[fullName];
        if (config.hasOwnProperty(fileName)) return +config[fileName];
        return 0;
    }
    return +config; // (false=0, true=1, number remains the same)
}

/**
 * Generates the file names for external data chunks.
 * @param {string} fullName The full ONNX file name (e.g., "model_quantized.onnx").
 * @param {number} numChunks The number of external data chunks.
 * @returns {string[]} Array of external data file names.
 */
export function getExternalDataChunkNames(fullName, numChunks) {
    const names = [];
    for (let i = 0; i < numChunks; ++i) {
        names.push(`${fullName}_data${i === 0 ? '' : '_' + i}`);
    }
    return names;
}

/**
 * Loads the core model file.
 *
 * @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
 * @param {string} fileName The base name of the model file (without suffix or extension).
 * @param {import('./hub.js').PretrainedModelOptions} options Additional options for loading the model.
 * @param {string} suffix The suffix to append to the file name (e.g., '_q4', '_quantized').
 * @returns {Promise<string|Uint8Array>} A Promise that resolves to the model file buffer or path.
 */
export async function getCoreModelFile(pretrained_model_name_or_path, fileName, options, suffix) {
    const baseName = `${fileName}${suffix}.onnx`;
    const fullPath = `${options.subfolder ?? ''}/${baseName}`;

    return await getModelFile(pretrained_model_name_or_path, fullPath, true, options, apis.IS_NODE_ENV);
}

/**
 * Loads external data files for a model.
 *
 * @param {string} pretrained_model_name_or_path The path to the directory containing the model files.
 * @param {string} fileName The base name of the model file (without suffix or extension).
 * @param {string} suffix The suffix to append to the file name (e.g., '_q4').
 * @param {import('./hub.js').PretrainedModelOptions} options Additional options for loading the model.
 * @param {import('./hub.js').ExternalData|Record<string, import('./hub.js').ExternalData>|undefined} use_external_data_format External data format configuration.
 * @param {any} [session_options] Optional session options that may contain externalData configuration.
 * @returns {Promise<Array<string|{path: string, data: Uint8Array}>>} A Promise that resolves to an array of external data files.
 */
export async function getModelDataFiles(
    pretrained_model_name_or_path,
    fileName,
    suffix,
    options,
    use_external_data_format,
    session_options = {},
) {
    const baseName = `${fileName}${suffix}.onnx`;
    const return_path = apis.IS_NODE_ENV;

    /** @type {Promise<string|{path: string, data: Uint8Array}>[]} */
    let externalDataPromises = [];

    const num_chunks = resolveExternalDataFormat(use_external_data_format, baseName, fileName);
    if (num_chunks > 0) {
        if (num_chunks > MAX_EXTERNAL_DATA_CHUNKS) {
            throw new Error(
                `The number of external data chunks (${num_chunks}) exceeds the maximum allowed value (${MAX_EXTERNAL_DATA_CHUNKS}).`,
            );
        }
        const chunkNames = getExternalDataChunkNames(baseName, num_chunks);
        for (const path of chunkNames) {
            const fullPath = `${options.subfolder ?? ''}/${path}`;
            externalDataPromises.push(
                new Promise(async (resolve, reject) => {
                    const data = await getModelFile(
                        pretrained_model_name_or_path,
                        fullPath,
                        true,
                        options,
                        return_path,
                    );
                    resolve(data instanceof Uint8Array ? { path, data } : path);
                }),
            );
        }
    } else if (session_options.externalData !== undefined) {
        externalDataPromises = session_options.externalData.map(async (ext) => {
            // if the external data is a string, fetch the file and replace the string with its content
            if (typeof ext.data === 'string') {
                const ext_buffer = await getModelFile(pretrained_model_name_or_path, ext.data, true, options);
                return { ...ext, data: ext_buffer };
            }
            return ext;
        });
    }

    return Promise.all(externalDataPromises);
}
