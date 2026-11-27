/**
 * @file Numerical helpers — softmax, dot product, cosine similarity, and the
 * typed-array utilities shared across the library.
 *
 * @module utils/maths
 */

/**
 * @typedef {Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float16Array | Float32Array | Float64Array} TypedArray
 * @typedef {BigInt64Array | BigUint64Array} BigTypedArray
 * @typedef {TypedArray | BigTypedArray} AnyTypedArray
 */

/**
 * @param {TypedArray} input
 */
export function interpolate_data(
    input,
    [in_channels, in_height, in_width],
    [out_height, out_width],
    mode = 'bilinear',
    align_corners = false,
) {
    // TODO use mode and align_corners

    // Output image dimensions
    const x_scale = out_width / in_width;
    const y_scale = out_height / in_height;

    // Output image
    // @ts-ignore
    const out_img = new input.constructor(out_height * out_width * in_channels);

    // Pre-calculate strides
    const inStride = in_height * in_width;
    const outStride = out_height * out_width;

    for (let i = 0; i < out_height; ++i) {
        for (let j = 0; j < out_width; ++j) {
            // Calculate output offset
            const outOffset = i * out_width + j;

            // Calculate input pixel coordinates
            const x = (j + 0.5) / x_scale - 0.5;
            const y = (i + 0.5) / y_scale - 0.5;

            // Calculate the four nearest input pixels
            // We also check if the input pixel coordinates are within the image bounds
            let x1 = Math.floor(x);
            let y1 = Math.floor(y);
            const x2 = Math.min(x1 + 1, in_width - 1);
            const y2 = Math.min(y1 + 1, in_height - 1);

            x1 = Math.max(x1, 0);
            y1 = Math.max(y1, 0);

            // Calculate the fractional distances between the input pixel and the four nearest pixels
            const s = x - x1;
            const t = y - y1;

            // Perform bilinear interpolation
            const w1 = (1 - s) * (1 - t);
            const w2 = s * (1 - t);
            const w3 = (1 - s) * t;
            const w4 = s * t;

            // Calculate the four nearest input pixel indices
            const yStride = y1 * in_width;
            const xStride = y2 * in_width;
            const idx1 = yStride + x1;
            const idx2 = yStride + x2;
            const idx3 = xStride + x1;
            const idx4 = xStride + x2;

            for (let k = 0; k < in_channels; ++k) {
                // Calculate channel offset
                const cOffset = k * inStride;

                out_img[k * outStride + outOffset] =
                    w1 * input[cOffset + idx1] +
                    w2 * input[cOffset + idx2] +
                    w3 * input[cOffset + idx3] +
                    w4 * input[cOffset + idx4];
            }
        }
    }

    return out_img;
}

/**
 * Helper method to permute a `AnyTypedArray` directly
 * @template {AnyTypedArray} T
 * @param {T} array
 * @param {number[]} dims
 * @param {number[]} axes
 * @returns {[T, number[]]} The permuted array and the new shape.
 */
export function permute_data(array, dims, axes) {
    // Calculate the new shape of the permuted array
    // and the stride of the original array
    const shape = new Array(axes.length);
    const stride = new Array(axes.length);

    for (let i = axes.length - 1, s = 1; i >= 0; --i) {
        stride[i] = s;
        shape[i] = dims[axes[i]];
        s *= shape[i];
    }

    // Precompute inverse mapping of stride
    const invStride = axes.map((_, i) => stride[axes.indexOf(i)]);

    // Create the permuted array with the new shape
    // @ts-ignore
    const permutedData = new array.constructor(array.length);

    // Permute the original array to the new array
    for (let i = 0; i < array.length; ++i) {
        let newIndex = 0;
        for (let j = dims.length - 1, k = i; j >= 0; --j) {
            newIndex += (k % dims[j]) * invStride[j];
            k = Math.floor(k / dims[j]);
        }
        permutedData[newIndex] = array[i];
    }

    return [permutedData, shape];
}

/**
 * Compute the softmax of an array of numbers.
 * @template {TypedArray|number[]} T
 * @param {T} arr The array of numbers to compute the softmax of.
 * @returns {T} The softmax array.
 */
export function softmax(arr) {
    // Compute the maximum value in the array
    const maxVal = max(arr)[0];

    // Compute the exponentials of the array values
    const exps = arr.map((x) => Math.exp(x - maxVal));

    // Compute the sum of the exponentials
    // @ts-ignore
    const sumExps = exps.reduce((acc, val) => acc + val, 0);

    // Compute the softmax values
    const softmaxArr = exps.map((x) => x / sumExps);

    return /** @type {T} */ (softmaxArr);
}

/**
 * Calculates the logarithm of the softmax function for the input array.
 * @template {TypedArray|number[]} T
 * @param {T} arr The input array to calculate the log_softmax function for.
 * @returns {T} The resulting log_softmax array.
 */
export function log_softmax(arr) {
    // Compute the maximum value in the array
    const maxVal = max(arr)[0];

    // Compute the sum of the exponentials
    let sumExps = 0;
    for (let i = 0; i < arr.length; ++i) {
        sumExps += Math.exp(arr[i] - maxVal);
    }

    // Compute the log of the sum
    const logSum = Math.log(sumExps);

    // Compute the softmax values
    const logSoftmaxArr = arr.map((x) => x - maxVal - logSum);

    return /** @type {T} */ (logSoftmaxArr);
}

/**
 * Calculates the dot product of two arrays.
 * @param {number[]} arr1 The first array.
 * @param {number[]} arr2 The second array.
 * @returns {number} The dot product of arr1 and arr2.
 */
export function dot(arr1, arr2) {
    let result = 0;
    for (let i = 0; i < arr1.length; ++i) {
        result += arr1[i] * arr2[i];
    }
    return result;
}

/**
 * Computes the cosine similarity between two arrays.
 *
 * @param {number[]} arr1 The first array.
 * @param {number[]} arr2 The second array.
 * @returns {number} The cosine similarity between the two arrays.
 */
export function cos_sim(arr1, arr2) {
    // Calculate dot product of the two arrays
    const dotProduct = dot(arr1, arr2);

    // Calculate the magnitude of the first array
    const magnitudeA = magnitude(arr1);

    // Calculate the magnitude of the second array
    const magnitudeB = magnitude(arr2);

    // Calculate the cosine similarity
    const cosineSimilarity = dotProduct / (magnitudeA * magnitudeB);

    return cosineSimilarity;
}

/**
 * Calculates the magnitude of a given array.
 * @param {number[]} arr The array to calculate the magnitude of.
 * @returns {number} The magnitude of the array.
 */
function magnitude(arr) {
    return Math.sqrt(arr.reduce((acc, val) => acc + val * val, 0));
}

/**
 * Returns the value and index of the minimum element in an array.
 * @template {number[]|bigint[]|AnyTypedArray} T
 * @param {T} arr array of numbers.
 * @returns {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} the value and index of the minimum element, of the form: [valueOfMin, indexOfMin]
 * @throws {Error} If array is empty.
 */
export function min(arr) {
    if (arr.length === 0) throw Error('Array must not be empty');
    let min = arr[0];
    let indexOfMin = 0;
    for (let i = 1; i < arr.length; ++i) {
        if (arr[i] < min) {
            min = arr[i];
            indexOfMin = i;
        }
    }
    return /** @type {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} */ ([min, indexOfMin]);
}

/**
 * Returns the value and index of the maximum element in an array.
 * @template {number[]|bigint[]|AnyTypedArray} T
 * @param {T} arr array of numbers.
 * @returns {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} the value and index of the maximum element, of the form: [valueOfMax, indexOfMax]
 * @throws {Error} If array is empty.
 */
export function max(arr) {
    if (arr.length === 0) throw Error('Array must not be empty');
    let max = arr[0];
    let indexOfMax = 0;
    for (let i = 1; i < arr.length; ++i) {
        if (arr[i] > max) {
            max = arr[i];
            indexOfMax = i;
        }
    }
    return /** @type {T extends bigint[]|BigTypedArray ? [bigint, number] : [number, number]} */ ([max, indexOfMax]);
}

/**
 * Performs median filter on the provided data. Padding is done by mirroring the data.
 * @param {AnyTypedArray} data The input array
 * @param {number} windowSize The window size
 */
export function medianFilter(data, windowSize) {
    if (windowSize % 2 === 0 || windowSize <= 0) {
        throw new Error('Window size must be a positive odd number');
    }

    // @ts-ignore
    const outputArray = new data.constructor(data.length);

    // @ts-ignore
    const buffer = new data.constructor(windowSize); // Reusable array for storing values

    const halfWindowSize = Math.floor(windowSize / 2);

    for (let i = 0; i < data.length; ++i) {
        let valuesIndex = 0;

        for (let j = -halfWindowSize; j <= halfWindowSize; ++j) {
            let index = i + j;
            if (index < 0) {
                index = Math.abs(index);
            } else if (index >= data.length) {
                index = 2 * (data.length - 1) - index;
            }

            buffer[valuesIndex++] = data[index];
        }

        buffer.sort();
        outputArray[i] = buffer[halfWindowSize];
    }

    return outputArray;
}

/**
 * Helper function to round a number to a given number of decimals
 * @param {number} num The number to round
 * @param {number} decimals The number of decimals
 * @returns {number} The rounded number
 */
export function round(num, decimals) {
    const pow = Math.pow(10, decimals);
    return Math.round(num * pow) / pow;
}

/**
 * Helper function to round a number to the nearest integer, with ties rounded to the nearest even number.
 * Also known as "bankers' rounding". This is the default rounding mode in python. For example:
 * 1.5 rounds to 2 and 2.5 rounds to 2.
 *
 * @param {number} x The number to round
 * @returns {number} The rounded number
 */
export function bankers_round(x) {
    const r = Math.round(x);
    const br = Math.abs(x) % 1 === 0.5 ? (r % 2 === 0 ? r : r - 1) : r;
    return br;
}

/**
 * Measures similarity between two temporal sequences (e.g., input audio and output tokens
 * to generate token-level timestamps).
 * @param {number[][]} matrix
 * @returns {number[][]}
 */
export function dynamic_time_warping(matrix) {
    const output_length = matrix.length;
    const input_length = matrix[0].length;

    const outputShape = [output_length + 1, input_length + 1];

    const cost = Array.from({ length: outputShape[0] }, () => Array(outputShape[1]).fill(Infinity));
    cost[0][0] = 0;

    const trace = Array.from({ length: outputShape[0] }, () => Array(outputShape[1]).fill(-1));

    for (let j = 1; j < outputShape[1]; ++j) {
        for (let i = 1; i < outputShape[0]; ++i) {
            const c0 = cost[i - 1][j - 1];
            const c1 = cost[i - 1][j];
            const c2 = cost[i][j - 1];

            let c, t;
            if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
            } else {
                c = c2;
                t = 2;
            }
            cost[i][j] = matrix[i - 1][j - 1] + c;
            trace[i][j] = t;
        }
    }

    for (let i = 0; i < outputShape[1]; ++i) {
        // trace[0, :] = 2
        trace[0][i] = 2;
    }
    for (let i = 0; i < outputShape[0]; ++i) {
        // trace[:, 0] = 1
        trace[i][0] = 1;
    }

    // backtrace
    let i = output_length;
    let j = input_length;
    let text_indices = [];
    let time_indices = [];
    while (i > 0 || j > 0) {
        text_indices.push(i - 1);
        time_indices.push(j - 1);

        switch (trace[i][j]) {
            case 0:
                --i;
                --j;
                break;
            case 1:
                --i;
                break;
            case 2:
                --j;
                break;
            default:
                throw new Error(
                    `Internal error in dynamic time warping. Unexpected trace[${i}, ${j}]. Please file a bug report.`,
                );
        }
    }

    text_indices.reverse();
    time_indices.reverse();

    return [text_indices, time_indices];
}

/**
 * Efficiently converts a Uint16Array of float16 values to a Float32Array.
 * This implementation uses a lazily initialized lookup table (LUT) for fast conversion.
 */
export const uint16_to_float32 = (function () {
    let float16LUT = null; // The Lookup Table

    return function (/** @type {Uint16Array} */ u16Array) {
        if (!float16LUT) {
            // Lazily initialize LUT
            float16LUT = new Float32Array(65536);
            const buffer = new ArrayBuffer(4);
            const u32 = new Uint32Array(buffer);
            const f32 = new Float32Array(buffer);

            for (let i = 0; i < float16LUT.length; ++i) {
                let outBits = 0;
                const sign = (i & 0x8000) << 16;
                const exp = (i & 0x7c00) >> 10;
                let mantissa = i & 0x03ff;

                if (exp === 0x1f) {
                    // Infinity or NaN
                    outBits = sign | 0x7f800000 | (mantissa << 13);
                } else if (exp === 0) {
                    // Zero or Subnormal
                    if (mantissa === 0) {
                        outBits = sign;
                    } else {
                        let renormExp = 113;
                        while ((mantissa & 0x0400) === 0) {
                            mantissa <<= 1;
                            --renormExp;
                        }
                        mantissa &= ~0x0400;
                        outBits = sign | (renormExp << 23) | (mantissa << 13);
                    }
                } else {
                    // Normal
                    outBits = sign | ((exp + 112) << 23) | (mantissa << 13);
                }

                u32[0] = outBits;
                float16LUT[i] = f32[0];
            }
        }

        const length = u16Array.length;
        const lut = float16LUT;
        const out = new Float32Array(length);
        for (let i = 0; i < length; ++i) {
            out[i] = lut[u16Array[i]];
        }

        return out;
    };
})();
