import { collapseCtcRepeats } from '../../src/utils/ctc.js';

const BLANK = 0;

describe('collapseCtcRepeats', () => {
    it('keeps a repeated letter when a blank separates the copies', () => {
        // "fellow": f e l blank l o w. Dropping the blank before the merge yields "felow".
        const ids = [6, 5, 12, BLANK, 12, 15, 23];
        expect(collapseCtcRepeats(ids, BLANK)).toEqual([6, 5, 12, 12, 15, 23]);
    });

    it('merges repeats that are not separated by a blank', () => {
        expect(collapseCtcRepeats([6, 6, 5, BLANK, BLANK, 12, 12], BLANK)).toEqual([6, 5, 12]);
    });

    it('drops a leading blank and matches a bigint blank id', () => {
        expect(collapseCtcRepeats([0n, 4n, 4n, 0n, 4n], 0)).toEqual([4n, 4n]);
    });

    it('leaves ids unchanged when the model has no blank id', () => {
        const ids = [1, 1, 2];
        expect(collapseCtcRepeats(ids, null)).toBe(ids);
    });
});
