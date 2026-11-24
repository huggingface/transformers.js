/**
 * Greedy CTC collapse. Repeated ids are merged, and the blank is dropped.
 * A blank between two copies of the same id keeps both. Decoding with
 * `skip_special_tokens` first removes the blank (the pad token) before that
 * merge, which deletes doubled letters.
 * @param {Array<number|bigint>} ids Argmax token ids, one per frame.
 * @param {number|bigint|null|undefined} blankId Blank id, usually `pad_token_id`.
 * @returns {Array<number|bigint>} Ids ready to decode.
 */
export function collapseCtcRepeats(ids, blankId) {
    if (blankId == null) {
        return ids;
    }
    // != so a numeric pad id still matches a bigint id.
    return ids.filter((id, i) => id != blankId && id != ids[i - 1]);
}
