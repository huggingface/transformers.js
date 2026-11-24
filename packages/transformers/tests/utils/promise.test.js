import { awaitWithBackground } from '../../src/utils/promise.js';

describe('awaitWithBackground', () => {
    it('surfaces the primary error and does not leave the background rejection unhandled', async () => {
        const unhandled = [];
        const onUnhandled = (error) => unhandled.push(error);
        process.on('unhandledRejection', onUnhandled);

        const background = Promise.reject(new Error('core failed'));
        await expect(awaitWithBackground(Promise.reject(new Error('external data failed')), background)).rejects.toThrow(
            'external data failed',
        );

        await new Promise((resolve) => setImmediate(resolve));
        process.off('unhandledRejection', onUnhandled);
        expect(unhandled).toEqual([]);
    });

    it('still rejects with the background error when the primary load succeeds', async () => {
        await expect(awaitWithBackground(Promise.resolve('external'), Promise.reject(new Error('core failed')))).rejects.toThrow(
            'core failed',
        );
    });

    it('returns both results when both loads succeed', async () => {
        await expect(awaitWithBackground(Promise.resolve('external'), Promise.resolve('weights'))).resolves.toEqual([
            'external',
            'weights',
        ]);
    });
});
