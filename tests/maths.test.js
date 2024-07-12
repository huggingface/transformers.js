
import { compare } from './test_utils.js';

import { getFile } from '../src/utils/hub.js';
import { FFT, medianFilter, bankers_round, log_softmax } from '../src/utils/maths.js';


const fft = (arr, complex = false) => {
    let output;
    let fft;
    if (complex) {
        fft = new FFT(arr.length / 2);
        output = new Float64Array(fft.outputBufferSize);
        fft.transform(output, arr);
    } else {
        fft = new FFT(arr.length);
        output = new Float64Array(fft.outputBufferSize);
        fft.realTransform(output, arr);
    }
    if (!fft.isPowerOfTwo) {
        output = output.slice(0, complex ? arr.length : 2 * arr.length);
    }
    return output;
}

const fftTestsData = await (await getFile('./tests/data/fft_tests.json')).json()

describe('Mathematical operations', () => {

    describe('bankers rounding', () => {
        it('should round up to nearest even', () => {
            expect(bankers_round(-0.5)).toBeCloseTo(0);
            expect(bankers_round(1.5)).toBeCloseTo(2);
            expect(bankers_round(19.5)).toBeCloseTo(20);
        });
        it('should round down to nearest even', () => {
            expect(bankers_round(-1.5)).toBeCloseTo(-2);
            expect(bankers_round(2.5)).toBeCloseTo(2);
            expect(bankers_round(18.5)).toBeCloseTo(18);
        });
    });

    describe('median filtering', () => {


        it('should compute median filter', async () => {
            const t1 = new Float32Array([5, 12, 2, 6, 3, 10, 9, 1, 4, 8, 11, 7]);
            const window = 3;

            const target = new Float32Array([12, 5, 6, 3, 6, 9, 9, 4, 4, 8, 8, 11]);

            const output = medianFilter(t1, window);
            compare(output, target, 1e-3);
        });


        // TODO add tests for errors
    });

    describe('FFT', () => {
        // Should match output of numpy fft
        it('should compute real FFT for power of two', () => {
            { // size = 4
                // np.fft.fft([1,2,3,4]) == array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
                const input = new Float32Array([1, 2, 3, 4]);
                const target = new Float32Array([10, 0, -2, 2, -2, 0, -2, -2]);

                const output = fft(input);
                compare(output, target, 1e-3);
            }

            { // size = 16
                // np.fft.fft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
                // == array([136. +0.j        ,  -8.+40.21871594j,  -8.+19.3137085j ,
                //            -8.+11.9728461j ,  -8. +8.j        ,  -8. +5.3454291j ,
                //            -8. +3.3137085j ,  -8. +1.59129894j,  -8. +0.j        ,
                //            -8. -1.59129894j,  -8. -3.3137085j ,  -8. -5.3454291j ,
                //            -8. -8.j        ,  -8.-11.9728461j ,  -8.-19.3137085j ,
                //            -8.-40.21871594j])
                const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
                const target = new Float32Array([136.0, 0.0, -8.0, 40.218715937006785, -8.0, 19.31370849898476, -8.0, 11.972846101323912, -8.0, 8.0, -8.0, 5.345429103354389, -8.0, 3.313708498984761, -8.0, 1.5912989390372658, -8.0, 0.0, -8.0, -1.5912989390372658, -8.0, -3.313708498984761, -8.0, -5.345429103354389, -8.0, -8.0, -8.0, -11.972846101323912, -8.0, -19.31370849898476, -8.0, -40.218715937006785]);

                const output = fft(input);
                compare(output, target, 1e-3);
            }
        });

        it('should compute real FFT for non-power of two', () => {
            { // size = 3
                // np.fft.fft([1,2,3]) == array([ 6. +0.j, -1.5+0.8660254j, -1.5-0.8660254j])
                const input = new Float32Array([1, 2, 3]);
                const target = new Float32Array([6, 0, -1.5, 0.8660254, -1.5, -0.8660254]);

                const output = fft(input);
                compare(output, target, 1e-3);
            }
        });

        it('should compute complex FFT for non-power of two', () => {
            { // size = 3
                // np.fft.fft([1+3j,2-2j,3+1j]) == array([ 6. +2.j, -4.09807621+4.3660254j, 1.09807621+2.6339746j])
                const input = new Float32Array([1, 3, 2, -2, 3, 1]);
                const target = new Float32Array([6, 2, -4.09807621, 4.3660254, 1.09807621, 2.6339746]);

                const output = fft(input, true);
                compare(output, target, 1e-3);
            }
        });

        it('should compute complex FFT for power of two', () => {
            { // size = 4
                // np.fft.fft([1+4j, 2-3j,3+2j, 4-1j]) == array([10. +2.j, -4. +4.j, -2.+10.j,  0. +0.j])
                const input = new Float32Array([1, 4, 2, -3, 3, 2, 4, -1]);
                const target = new Float32Array([10, 2, -4, 4, -2, 10, 0, 0]);

                const output = fft(input, true);
                compare(output, target, 1e-3);
            }
        });
    })

    describe('FFT (dynamic)', () => {
        // Should match output of numpy fft
        for (const [name, test] of Object.entries(fftTestsData)) {
            // if (test.input.length > 5) continue;
            it(name, () => {
                const output = fft(test.input, test.complex);

                if (output.map((v, i) => Math.abs(v - test.output[i])).some(v => v > 1e-4)) {
                    console.log('input', test.input)
                    console.log('output', output)
                    console.log('target', test.output)
                }
                compare(output, test.output, 1e-4);

            });
        }
    });

    describe('log softmax', () => {
        // Should match output of scipy log_softmax
        it('should compute log softmax correctly for usual values', () => {
            const input = [0, 1, 2, 3];
            const expected = [-3.4401896985611953, -2.4401896985611953, -1.4401896985611953, -0.44018969856119533];
            const output = log_softmax(input);
            compare(output, expected, 1e-13);
        });

        it('should compute log softmax correctly for values with large differences', () => {
            const input = [1000, 1];
            const expected = [0, -999];
            const output = log_softmax(input);
            compare(output, expected, 1e-13);
        });
    });
});
