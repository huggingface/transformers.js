/**
 * Select the highest scoring tokens without sorting or allocating an object
 * for every vocabulary entry. Ties keep ascending token order.
 * @param {Float32Array} log_probs
 * @param {number} beam_score
 * @param {number} beam_idx
 * @param {number} count
 * @param {Map<bigint, number>} [token_counts]
 * @param {number} [diversity_penalty]
 * @returns {{score: number, token: bigint, beam_idx: number}[]}
 * @internal
 */
export function selectBeamCandidates(log_probs, beam_score, beam_idx, count, token_counts, diversity_penalty = 0) {
    count = Math.min(log_probs.length, Math.max(0, Math.floor(count)));
    if (count === 0) return [];
    const penalties = new Map();
    if (token_counts && diversity_penalty !== 0) {
        for (const [token, frequency] of token_counts) penalties.set(Number(token), frequency * diversity_penalty);
    }
    if (count === log_probs.length) {
        return Array.from(log_probs, (score, token) => ({
            score: beam_score + score - (penalties.get(token) ?? 0),
            token: BigInt(token),
            beam_idx,
        })).sort((a, b) => b.score - a.score);
    }
    /** @type {{score: number, token: number}[]} */
    const heap = [];
    const worse = (a, b) => a.score < b.score || (a.score === b.score && a.token > b.token);
    for (let token = 0; token < log_probs.length; ++token) {
        const score = beam_score + log_probs[token] - (penalties.get(token) ?? 0);
        if (heap.length < count) {
            let child = heap.length;
            heap.push({ score, token });
            while (child > 0) {
                const parent = (child - 1) >> 1;
                if (!worse(heap[child], heap[parent])) break;
                [heap[child], heap[parent]] = [heap[parent], heap[child]];
                child = parent;
            }
        } else if (score > heap[0].score) {
            heap[0] = { score, token };
            let parent = 0;
            while (parent * 2 + 1 < heap.length) {
                let child = parent * 2 + 1;
                if (child + 1 < heap.length && worse(heap[child + 1], heap[child])) ++child;
                if (!worse(heap[child], heap[parent])) break;
                [heap[child], heap[parent]] = [heap[parent], heap[child]];
                parent = child;
            }
        }
    }
    return heap
        .sort((a, b) => b.score - a.score || a.token - b.token)
        .map(({ score, token }) => ({ score, token: BigInt(token), beam_idx }));
}
