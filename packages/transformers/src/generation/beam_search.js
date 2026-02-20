/**
 * @module generation/beam_search
 */

/**
 * Stores completed beam search hypotheses for a single batch element.
 */
export class BeamHypotheses {
    /**
     * @param {number} num_beams Number of beams.
     * @param {number} length_penalty Exponential penalty to the length.
     * @param {boolean|"never"} early_stopping Whether to stop when enough hypotheses are finished.
     */
    constructor(num_beams, length_penalty = 1.0, early_stopping = false) {
        this.num_beams = num_beams;
        this.length_penalty = length_penalty;
        this.early_stopping = early_stopping;

        /** @type {{score: number, tokens: bigint[]}[]} */
        this.beams = [];
        this.worst_score = 1e9;
    }

    get length() {
        return this.beams.length;
    }

    /**
     * Add a new hypothesis to the list.
     * @param {number} sum_logprobs Sum of log probabilities of the hypothesis.
     * @param {bigint[]} tokens The token ids of the hypothesis.
     */
    add(sum_logprobs, tokens) {
        const score = sum_logprobs / (tokens.length ** this.length_penalty);
        if (this.beams.length < this.num_beams || score > this.worst_score) {
            this.beams.push({ score, tokens });
            if (this.beams.length > this.num_beams) {
                // Remove worst hypothesis
                let worst_idx = 0;
                for (let i = 1; i < this.beams.length; ++i) {
                    if (this.beams[i].score < this.beams[worst_idx].score) {
                        worst_idx = i;
                    }
                }
                this.beams.splice(worst_idx, 1);
            }
            this.worst_score = this.beams.length === this.num_beams
                ? Math.min(...this.beams.map(b => b.score))
                : -1e9;
        }
    }

    /**
     * Check whether adding more beams can possibly improve the hypotheses.
     * @param {number} best_sum_logprobs Best sum of log probs among active beams.
     * @param {number} cur_len Current length of generated tokens.
     * @returns {boolean}
     */
    is_done(best_sum_logprobs, cur_len) {
        if (this.beams.length < this.num_beams) return false;

        if (this.early_stopping === true) {
            return true;
        } else if (this.early_stopping === 'never') {
            return false;
        } else {
            // Heuristic: check if the best possible score for the next step
            // could beat the worst completed hypothesis
            const highest_attainable_score = best_sum_logprobs / (cur_len ** this.length_penalty);
            return this.worst_score >= highest_attainable_score;
        }
    }
}

/**
 * Implements beam search scoring and beam management.
 */
export class BeamSearchScorer {
    /**
     * @param {number} batch_size
     * @param {number} num_beams
     * @param {Object} options
     * @param {number} [options.length_penalty]
     * @param {boolean|"never"} [options.early_stopping]
     * @param {number} [options.num_return_sequences]
     * @param {number|number[]|null} [options.eos_token_id]
     * @param {number|null} [options.pad_token_id]
     */
    constructor(batch_size, num_beams, {
        length_penalty = 1.0,
        early_stopping = false,
        num_return_sequences = 1,
        eos_token_id = null,
        pad_token_id = null,
    } = {}) {
        this.batch_size = batch_size;
        this.num_beams = num_beams;
        this.length_penalty = length_penalty;
        this.early_stopping = early_stopping;
        this.num_return_sequences = num_return_sequences;
        this.eos_token_ids = eos_token_id === null ? []
            : (Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id]);
        this.pad_token_id = pad_token_id ?? 0;

        if (num_return_sequences > num_beams) {
            throw new Error(
                `num_return_sequences (${num_return_sequences}) must be <= num_beams (${num_beams}).`,
            );
        }

        /** @type {BeamHypotheses[]} */
        this._beam_hyps = Array.from(
            { length: batch_size },
            () => new BeamHypotheses(num_beams, length_penalty, early_stopping),
        );
        /** @type {boolean[]} */
        this._done = new Array(batch_size).fill(false);
    }

    get is_done() {
        return this._done.every(Boolean);
    }

    /**
     * Process the current beam candidates and select the next set of beams.
     *
     * @param {bigint[][]} all_input_ids All sequences so far, shape [batch_size * num_beams, seq_len].
     * @param {number[]} beam_scores Cumulative scores, shape [batch_size * num_beams].
     * @param {bigint[]} next_tokens Candidate tokens, shape [batch_size * 2 * num_beams].
     * @param {number[]} next_indices Which beam each candidate came from (relative to batch group), shape [batch_size * 2 * num_beams].
     * @param {number[]} next_scores New cumulative scores for candidates, shape [batch_size * 2 * num_beams].
     * @returns {{ next_beam_scores: number[], next_beam_tokens: bigint[], next_beam_indices: number[] }}
     */
    process(all_input_ids, beam_scores, next_tokens, next_indices, next_scores) {
        const cur_len = all_input_ids[0].length;
        const total_beams = this.batch_size * this.num_beams;

        const next_beam_scores = new Array(total_beams).fill(0);
        const next_beam_tokens = new Array(total_beams).fill(0n);
        const next_beam_indices = new Array(total_beams).fill(0);

        for (let batch_idx = 0; batch_idx < this.batch_size; ++batch_idx) {
            if (this._done[batch_idx]) {
                // Pad finished batches
                for (let beam_idx = 0; beam_idx < this.num_beams; ++beam_idx) {
                    const flat_idx = batch_idx * this.num_beams + beam_idx;
                    next_beam_scores[flat_idx] = 0;
                    next_beam_tokens[flat_idx] = BigInt(this.pad_token_id);
                    next_beam_indices[flat_idx] = batch_idx * this.num_beams;
                }
                continue;
            }

            let beam_idx = 0;
            const num_candidates = 2 * this.num_beams;
            for (let j = 0; j < num_candidates; ++j) {
                const cand_idx = batch_idx * num_candidates + j;
                const beam_token = next_tokens[cand_idx];
                const beam_score = next_scores[cand_idx];
                const beam_source = next_indices[cand_idx]; // relative to batch
                const abs_beam_source = batch_idx * this.num_beams + beam_source;

                const is_eos = this.eos_token_ids.some(id => BigInt(id) === beam_token);

                if (is_eos) {
                    // Add completed hypothesis
                    const hypothesis = [...all_input_ids[abs_beam_source], beam_token];
                    this._beam_hyps[batch_idx].add(beam_score, hypothesis);
                } else {
                    // Add to next active beams
                    const out_idx = batch_idx * this.num_beams + beam_idx;
                    next_beam_scores[out_idx] = beam_score;
                    next_beam_tokens[out_idx] = beam_token;
                    next_beam_indices[out_idx] = abs_beam_source;
                    beam_idx++;
                }

                if (beam_idx === this.num_beams) break;
            }

            // If we couldn't fill all beams (too many EOS), pad with last valid
            if (beam_idx < this.num_beams) {
                const last_valid = batch_idx * this.num_beams + Math.max(0, beam_idx - 1);
                for (; beam_idx < this.num_beams; ++beam_idx) {
                    const out_idx = batch_idx * this.num_beams + beam_idx;
                    next_beam_scores[out_idx] = next_beam_scores[last_valid] ?? 0;
                    next_beam_tokens[out_idx] = next_beam_tokens[last_valid] ?? 0n;
                    next_beam_indices[out_idx] = next_beam_indices[last_valid] ?? (batch_idx * this.num_beams);
                }
            }

            // Check if done for this batch using next-step scores
            const start = batch_idx * this.num_beams;
            const end = start + this.num_beams;
            const best_sum_logprobs = Math.max(...next_beam_scores.slice(start, end));
            this._done[batch_idx] = this._beam_hyps[batch_idx].is_done(best_sum_logprobs, cur_len + 1);
        }

        return { next_beam_scores, next_beam_tokens, next_beam_indices };
    }

    /**
     * Finalize: select best hypotheses.
     * @param {bigint[][]} all_input_ids Final sequences, shape [batch_size * num_beams, seq_len].
     * @param {number[]} beam_scores Final cumulative scores.
     * @returns {bigint[][]} Best sequences, shape [batch_size * num_return_sequences, seq_len].
     */
    finalize(all_input_ids, beam_scores) {
        return this.finalize_with_scores(all_input_ids, beam_scores).map((x) => x.tokens);
    }

    /**
     * Finalize: select best hypotheses and return scores.
     * @param {bigint[][]} all_input_ids Final sequences, shape [batch_size * num_beams, seq_len].
     * @param {number[]} beam_scores Final cumulative scores.
     * @returns {{tokens: bigint[], score: number}[]} Best sequences with scores.
     */
    finalize_with_scores(all_input_ids, beam_scores) {
        // For each batch, ensure we have enough hypotheses
        for (let batch_idx = 0; batch_idx < this.batch_size; ++batch_idx) {
            const hyps = this._beam_hyps[batch_idx];
            if (hyps.length < this.num_beams) {
                for (let beam_idx = 0; beam_idx < this.num_beams; ++beam_idx) {
                    const flat_idx = batch_idx * this.num_beams + beam_idx;
                    hyps.add(beam_scores[flat_idx], all_input_ids[flat_idx]);
                }
            }
        }

        // Select top num_return_sequences per batch
        const results = [];
        for (let batch_idx = 0; batch_idx < this.batch_size; ++batch_idx) {
            const sorted = [...this._beam_hyps[batch_idx].beams]
                .sort((a, b) => b.score - a.score);
            for (let i = 0; i < this.num_return_sequences; ++i) {
                results.push(sorted[i]);
            }
        }
        return results;
    }
}
