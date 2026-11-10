import { BeamHypotheses, BeamSearchScorer } from "../../src/generation/beam_search.js";

describe("Beam hypothesis scoring", () => {
  it("normalizes by generated length rather than prompt length", () => {
    const scorer = new BeamSearchScorer(1, 2, { decoder_prompt_len: 3, early_stopping: "never" });
    scorer.process(
      [
        [10n, 11n, 12n],
        [10n, 11n, 12n],
      ],
      [0, -1e9],
      [2n, 3n, 4n, 5n],
      [0, 0, 0, 0],
      [-1, -2, -3, -4],
    );
    const result = scorer.finalize_with_scores(
      [
        [10n, 11n, 12n, 2n],
        [10n, 11n, 12n, 3n],
      ],
      [-1, -2],
    );
    expect(result[0].score).toBe(-1);
  });

  it("compares active beams with a full completed pool at the length limit", () => {
    const scorer = new BeamSearchScorer(1, 2, { early_stopping: "never", length_penalty: 0 });
    scorer._beam_hyps[0].add(-5, [1n, 0n]);
    scorer._beam_hyps[0].add(-6, [2n, 0n]);
    expect(
      scorer.finalize(
        [
          [1n, 3n],
          [2n, 4n],
        ],
        [-1, -2],
      ),
    ).toEqual([[1n, 3n]]);
  });

  it("does not resurrect padded beams from a finished batch", () => {
    const scorer = new BeamSearchScorer(1, 2, { early_stopping: true, length_penalty: 0 });
    scorer._beam_hyps[0].add(-5, [1n, 0n]);
    scorer._beam_hyps[0].add(-6, [2n, 0n]);
    scorer._done[0] = true;
    expect(
      scorer.finalize(
        [
          [1n, 9n],
          [2n, 9n],
        ],
        [0, 0],
      ),
    ).toEqual([[1n, 0n]]);
  });

  it("only completes EOS candidates within the top beam ranks", () => {
    const scorer = new BeamSearchScorer(1, 2, { eos_token_id: 0 });
    scorer.process([[1n], [2n]], [0, 0], [0n, 3n, 0n, 4n], [0, 0, 1, 1], [-1, -2, -3, -4]);
    expect(scorer._beam_hyps[0].beams.map((beam) => beam.tokens)).toEqual([[1n, 0n]]);
  });

  it("handles more than two EOS ids without inventing active tokens", () => {
    const scorer = new BeamSearchScorer(1, 2, { eos_token_id: [0, 1, 2] });
    const result = scorer.process([[9n], [9n]], [0, -1e9], [0n, 1n, 2n, 3n, 4n, 5n, 6n, 7n], Array(8).fill(0), [-1, -2, -3, -4, -5, -6, -7, -8]);
    expect(result.next_beam_tokens).toEqual([3n, 4n]);
  });

  it('stops "never" when no completion can improve the score', () => {
    const hypotheses = new BeamHypotheses(2, 1, "never", 5);
    hypotheses.add(-1, [1n, 0n]);
    hypotheses.add(-2, [2n, 0n]);
    expect(hypotheses.is_done(-10, 3)).toBe(true);
    expect(hypotheses.is_done(-1, 3)).toBe(false);
  });
});

describe("Exhausted beam candidates", () => {
  it("returns a real EOS hypothesis when no continuation exists", () => {
    const scorer = new BeamSearchScorer(1, 2, { eos_token_id: [1, 2], decoder_prompt_len: 1, max_length: 2 });
    const next = scorer.process([[9n], [9n]], [0, -1e9], [1n, 2n, 1n, 2n], [0, 0, 1, 1], [-0.3, -1.3, -1e9, -1e9]);
    expect(scorer.is_done).toBe(true);
    expect(next.next_beam_scores).toEqual([-Infinity, -Infinity]);
    expect(
      scorer.finalize(
        [
          [9n, 0n],
          [9n, 0n],
        ],
        next.next_beam_scores,
      ),
    ).toEqual([[9n, 1n]]);
  });
});
